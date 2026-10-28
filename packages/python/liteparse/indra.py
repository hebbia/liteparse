"""`indra_ocr`-backed :class:`LiteParse` OCR engine (``ocr_engine=...``).

Lets Indra supply the text runs for pages LiteParse flags as needing OCR, so
the same spatial projection / layout model that handles native text also lays
out the recognized lines — a single LiteParse pass with no ocrmypdf
round-trip and no second pipeline::

    from liteparse import LiteParse
    from liteparse.indra import IndraOcrEngine

    parser = LiteParse(ocr_engine=IndraOcrEngine(), output_format="markdown")
    result = parser.parse("scanned.pdf")

Requires ``indra_ocr`` (and its ``numpy``/``onnxruntime`` deps) to be
installed; imported lazily so LiteParse works without them.
"""

from __future__ import annotations

from typing import Any, List, Optional

__all__ = ["IndraOcrEngine"]


class IndraOcrEngine:
    """OcrEngine callback that runs :mod:`indra_ocr` line detection +
    recognition on each raster LiteParse renders for OCR.

    Follows the ``ocr_engine`` contract documented on
    :meth:`LiteParse.__init__`: called as
    ``recognize(data, width, height, channels, dpi, language)`` with tightly
    packed pixels (``prefers_grayscale`` makes them 1 byte/px luma) and
    returns ``{"text", "bbox", "confidence"}`` dicts per detected line — the
    same granularity Tesseract emits, so downstream merging treats both
    engines alike.

    ``prefers_grayscale = True``: Indra's detector binarizes internally, so
    the cheaper 1-channel raster is all it needs.
    """

    name = "indra"
    prefers_grayscale = True

    def __init__(
        self,
        model_path: Optional[str] = None,
        threads: int = 4,
        providers: Optional[List[str]] = None,
        math: bool = False,
    ) -> None:
        """Load Indra's ONNX models (same options as ``indra_ocr.load``).

        Args:
            model_path: Directory with the ONNX models; defaults to the
                models bundled with ``indra_ocr``.
            threads: ONNX Runtime inference threads.
            providers: ONNX Runtime execution providers (default CPU).
            math: Read math lines as LaTeX (needs the math models; slower).
        """
        from indra_ocr.models import model_dir
        from indra_ocr.pipeline import OrtOCR

        # furniture=True mirrors the production ``indra_ocr.load(...)`` call:
        # page headers/footers stay in the recognized output so LiteParse's
        # own header/footer stripping decides what to drop.
        self._ocr = OrtOCR(
            model_dir(model_path, math=math),
            threads=threads,
            providers=providers,
            furniture=True,
            math=math,
        )

    def recognize(
        self,
        data: bytes,
        width: int,
        height: int,
        channels: int,
        dpi: float,
        language: str,
    ) -> List[dict]:
        import numpy as np

        buf = np.frombuffer(data, dtype=np.uint8)
        if channels == 1:
            gray = buf.reshape(height, width)
        else:
            # Indra detects on grayscale (ITU-R 601 luma).
            px = buf.reshape(height, width, channels)[..., :3].astype(np.float32)
            gray = (px @ [0.299, 0.587, 0.114]).astype(np.uint8)

        boxes, _picture_boxes = self._ocr.detect_all(gray)
        if not len(boxes):
            return []
        return [
            {
                "text": text,
                "bbox": [float(v) for v in box[:4]],
                "confidence": float(conf),
            }
            for box, (text, conf, _tokens) in zip(boxes, self._ocr.recognize(gray, boxes))
            if text
        ]

    # The OcrEngine protocol accepts a callable or an object exposing
    # `recognize`; being callable too keeps duck-typed engines happy.
    def __call__(self, *args: Any, **kwargs: Any) -> List[dict]:
        return self.recognize(*args, **kwargs)
