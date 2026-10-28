"""Tests for the ``ocr_engine`` callback and the ``liteparse.indra`` adapter."""

from pathlib import Path

import pytest

from liteparse import LiteParse

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE_ONLY_PDF = REPO_ROOT / "integration_tests_data" / "image_only_receipt.pdf"


@pytest.fixture
def image_only_pdf() -> Path:
    """A one-page PDF that is a raster scan: no text layer -> needs_ocr."""
    if not IMAGE_ONLY_PDF.exists():
        pytest.skip(f"Test document not found: {IMAGE_ONLY_PDF}")
    return IMAGE_ONLY_PDF


class RecordingEngine:
    """Captures each raster it's asked to recognize."""

    name = "recording"
    prefers_grayscale = True

    def __init__(self, result):
        self._result = result
        self.calls = []

    def recognize(self, data, width, height, channels, dpi, language):
        self.calls.append(
            {
                "data_len": len(data),
                "width": width,
                "height": height,
                "channels": channels,
                "dpi": dpi,
                "language": language,
            }
        )
        return self._result


LINE = {"text": "Hello scanned page", "bbox": [10.0, 20.0, 200.0, 40.0]}


class TestOcrEngineCallback:
    def test_engine_text_runs_reach_output(self, image_only_pdf):
        engine = RecordingEngine([LINE])
        parser = LiteParse(ocr_engine=engine, output_format="markdown")
        result = parser.parse(str(image_only_pdf))
        assert "Hello scanned page" in result.text
        assert engine.calls, "engine was never invoked"

    def test_raster_is_grayscale_when_engine_prefers_it(self, image_only_pdf):
        engine = RecordingEngine([LINE])
        LiteParse(ocr_engine=engine).parse(str(image_only_pdf))
        for call in engine.calls:
            assert call["channels"] == 1
            assert call["data_len"] == call["width"] * call["height"]

    def test_engine_implies_ocr_enabled(self, image_only_pdf):
        # No ocr_enabled argument — supplying an engine turns OCR on.
        engine = RecordingEngine([LINE])
        result = LiteParse(ocr_engine=engine).parse(str(image_only_pdf))
        assert engine.calls
        assert "Hello scanned page" in result.text

    def test_explicit_ocr_disabled_wins(self, image_only_pdf):
        engine = RecordingEngine([LINE])
        LiteParse(ocr_engine=engine, ocr_enabled=False).parse(str(image_only_pdf))
        assert not engine.calls

    def test_tuple_results_accepted(self, image_only_pdf):
        engine = RecordingEngine([("Hello scanned page", [10.0, 20.0, 200.0, 40.0], 0.9)])
        result = LiteParse(ocr_engine=engine).parse(str(image_only_pdf))
        assert "Hello scanned page" in result.text

    def test_empty_result_falls_back_to_empty_page(self, image_only_pdf):
        engine = RecordingEngine([])
        result = LiteParse(ocr_engine=engine, continue_on_page_error=True).parse(
            str(image_only_pdf)
        )
        assert not result.page_errors

    def test_non_callable_engine_rejected(self):
        with pytest.raises(TypeError):
            LiteParse(ocr_engine=object())

    def test_engine_rejected_with_pool_size(self):
        with pytest.raises(ValueError, match="pool_size"):
            LiteParse(ocr_engine=lambda *_: [], pool_size=1)

    def test_callback_error_does_not_crash_parse(self, image_only_pdf):
        def boom(*_args):
            raise RuntimeError("engine exploded")

        result = LiteParse(
            ocr_engine=boom, ocr_failure_fatal=False, continue_on_page_error=True
        ).parse(str(image_only_pdf))
        assert result.num_pages == 1


class TestIndraOcrEngine:
    def test_real_indra_ocr_runs_through_layout(self, image_only_pdf):
        pytest.importorskip("indra_ocr", reason="indra_ocr not installed")
        from liteparse.indra import IndraOcrEngine

        engine = IndraOcrEngine()
        result = LiteParse(ocr_engine=engine, output_format="markdown").parse(
            str(image_only_pdf)
        )
        # A scanned receipt produces real text; exact words vary with the model.
        assert len(result.pages[0].text.strip()) > 20

    def test_indra_engine_contract(self):
        pytest.importorskip("indra_ocr", reason="indra_ocr not installed")
        from liteparse.indra import IndraOcrEngine

        engine = IndraOcrEngine()
        assert engine.name == "indra"
        assert engine.prefers_grayscale is True
        # A blank raster yields no lines.
        assert engine.recognize(bytes(64 * 64), 64, 64, 1, 200.0, "eng") == []
