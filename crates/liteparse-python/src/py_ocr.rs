//! `OcrEngine` backed by a Python callable — the analogue of the WASM
//! `ocrEngine` callback for the Python bindings.
//!
//! The callable receives the raw page pixels the Rust side rendered for OCR
//! and returns word- or line-level results that merge into the document's
//! text runs exactly like Tesseract or HTTP-server output would:
//!
//! ```python
//! def recognize(data: bytes, width: int, height: int,
//!               channels: int, dpi: float, language: str):
//!     # -> [{"text": str, "bbox": [x1, y1, x2, y2],
//!     #      "confidence": float,                      # optional
//!     #      "polygon": [[x, y], ...]}, ...]           # optional
//!     # or (text, bbox) / (text, bbox, confidence) / (text, bbox, confidence, polygon) tuples
//! ```
//!
//! `data` is tightly packed pixels: `channels` 1 = 1 byte/px luma,
//! 3 = interleaved RGB, in row order. Setting `prefers_grayscale = True` on
//! the callable object makes the renderer emit 1-channel data (cheaper, and
//! what engines that binarize internally expect).

use std::pin::Pin;
use std::sync::Arc;

use liteparse::ocr::{OcrEngine, OcrOptions, OcrResult};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

/// How many channels the raster carries, inferred from the buffer size.
fn channels(image_data: &[u8], width: u32, height: u32) -> u32 {
    let pixels = image_data.len() as u64;
    let cells = width as u64 * height as u64;
    if cells == 0 {
        return 0;
    }
    (pixels / cells) as u32
}

fn err(msg: impl Into<String>) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(std::io::Error::new(std::io::ErrorKind::Other, msg.into()))
}

/// Extract one OCR item from either a mapping
/// ({"text", "bbox", "confidence"?, "polygon"?}) or a sequence
/// (text, bbox[, confidence[, polygon]]).
fn extract_item(item: &Bound<'_, PyAny>) -> PyResult<OcrResult> {
    if let Ok(dict) = item.cast::<PyDict>() {
        let text: String = dict
            .get_item("text")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("ocr result item missing 'text'"))?
            .extract()?;
        let bbox: [f32; 4] = dict
            .get_item("bbox")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("ocr result item missing 'bbox'"))?
            .extract()?;
        let confidence: f32 = dict
            .get_item("confidence")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(1.0);
        let polygon: Option<[[f32; 2]; 4]> =
            dict.get_item("polygon")?.map(|v| v.extract()).transpose()?;
        return Ok(OcrResult {
            text,
            bbox,
            confidence,
            polygon,
        });
    }
    if let Ok(tuple) = item.cast::<PyTuple>() {
        let len = tuple.len();
        if len < 2 || len > 4 {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "ocr result tuple must have 2-4 items, got {len}"
            )));
        }
        let text: String = tuple.get_item(0)?.extract()?;
        let bbox: [f32; 4] = tuple.get_item(1)?.extract()?;
        let confidence: f32 = if len >= 3 {
            tuple.get_item(2)?.extract()?
        } else {
            1.0
        };
        let polygon: Option<[[f32; 2]; 4]> = if len >= 4 {
            tuple.get_item(3)?.extract()?
        } else {
            None
        };
        return Ok(OcrResult {
            text,
            bbox,
            confidence,
            polygon,
        });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "ocr result items must be dicts or (text, bbox[, confidence[, polygon]]) tuples",
    ))
}

pub struct PyOcrEngine {
    callback: Py<PyAny>,
    engine_name: String,
    grayscale: bool,
}

impl PyOcrEngine {
    /// `callback` must be callable; `name` and `prefers_grayscale` attributes
    /// on it (as on [`liteparse.indra.IndraOcrEngine`]) override the defaults
    /// "python" and `false`.
    pub fn new(callback: Py<PyAny>, py: Python<'_>) -> PyResult<Self> {
        let obj = callback.bind(py);
        if !obj.hasattr("recognize")? && !obj.is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "ocr_engine must be a callable, or an object with a `recognize` method",
            ));
        }
        let engine_name: String = obj
            .getattr("name")
            .and_then(|v| v.extract())
            .unwrap_or_else(|_| "python".to_string());
        let grayscale: bool = obj
            .getattr("prefers_grayscale")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        Ok(Self {
            callback,
            engine_name,
            grayscale,
        })
    }

    fn call_recognize(
        callback: &Py<PyAny>,
        image_data: &[u8],
        width: u32,
        height: u32,
        options: &OcrOptions,
    ) -> PyResult<Vec<OcrResult>> {
        Python::attach(|py| {
            let obj = callback.bind(py);
            let target = if let Ok(rec) = obj.getattr("recognize") {
                rec
            } else {
                obj.clone().into_any()
            };
            let n_channels = channels(image_data, width, height);
            let result = target.call1((
                PyBytes::new(py, image_data),
                width,
                height,
                n_channels,
                options.dpi,
                options.language.as_str(),
            ))?;
            result
                .try_iter()?
                .map(|item| extract_item(&item?))
                .collect()
        })
    }
}

impl OcrEngine for PyOcrEngine {
    fn name(&self) -> &str {
        &self.engine_name
    }

    fn prefers_grayscale(&self) -> bool {
        self.grayscale
    }

    fn recognize<'a, 'b: 'a, 'c: 'a>(
        &'a self,
        image_data: &'c [u8],
        width: u32,
        height: u32,
        options: &'b OcrOptions,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Vec<OcrResult>, Box<dyn std::error::Error + Send + Sync>>>
                + Send
                + '_,
        >,
    > {
        // Python calls can be slow (e.g. ONNX inference); the caller already
        // wraps engines in spawn_blocking, so run the GIL-bound call inline.
        Box::pin(async move {
            Self::call_recognize(&self.callback, image_data, width, height, options)
                .map_err(|e| err(format!("ocr_engine callback failed: {e}")))
        })
    }
}

/// Convenience for `LiteParse::new`: build the shared engine when a callback
/// object was provided.
pub fn wrap(callback: Py<PyAny>, py: Python<'_>) -> PyResult<Arc<dyn OcrEngine>> {
    Ok(Arc::new(PyOcrEngine::new(callback, py)?))
}
