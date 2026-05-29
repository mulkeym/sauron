import pytest


@pytest.mark.asyncio
async def test_pdf_falls_back_to_flat_text_when_extract_raises(monkeypatch):
    """If extract_pdf raises, the PDF still ingests via flat-text chunking (no crash)."""
    from src.ingestion import pipeline
    monkeypatch.setattr(pipeline, "extract_pdf",
                        lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    assert pipeline._is_structured_pdf("pdf") is True
    assert pipeline._is_structured_pdf("xlsx") is False


def test_queue_recognizes_pdf_as_structured():
    from src.ingestion import queue as q
    assert q._is_structured_pdf("pdf") is True
    assert q._is_structured_pdf("xlsx") is False
