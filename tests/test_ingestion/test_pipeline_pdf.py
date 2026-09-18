import pytest


@pytest.mark.asyncio
async def test_worker_crash_does_not_fall_back_to_api_parser(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock, MagicMock
    from src.ingestion import pipeline
    from src.ingestion.isolation import ExtractionWorkerError
    worker = AsyncMock(side_effect=ExtractionWorkerError("SIGSEGV"))
    monkeypatch.setattr(pipeline, "extract_in_worker", worker)
    vector_store, metadata_store = MagicMock(), AsyncMock()
    with pytest.raises(ExtractionWorkerError, match="SIGSEGV"):
        await pipeline.ingest_document(tmp_path / "bad.pdf", [], "tester", vector_store, metadata_store)
    vector_store.upsert.assert_not_called()
    metadata_store.add_document.assert_not_awaited()


def test_queue_recognizes_pdf_as_structured():
    from src.ingestion import queue as q
    assert q._is_structured_pdf("pdf") is True
    assert q._is_structured_pdf("xlsx") is False
