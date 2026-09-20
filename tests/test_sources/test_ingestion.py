import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import settings
from src.db.metadata import MetadataStore
from src.ingestion.queue import IngestQueue, IngestStep
from src.sources.storage import OriginalStore


@pytest.mark.asyncio
async def test_queue_retains_original_before_deleting_upload(tmp_path, monkeypatch):
    from src.ingestion.prepared import prepare_document

    monkeypatch.setattr(settings, "source_originals_dir", str(tmp_path / "originals"))
    monkeypatch.setattr("src.ingestion.isolation.extract_in_worker", prepare_document)
    monkeypatch.setattr(
        "src.ingestion.embedder.embed_texts",
        lambda texts, *args, **kwargs: [[0.1] * 1024 for _ in texts],
    )
    monkeypatch.setattr(
        "src.generation.llm_client.generate", lambda **kwargs: "A test summary"
    )
    monkeypatch.setattr("src.knowledge.graph_rag.insert_document", AsyncMock())
    source = Path(__file__).parents[2] / "test_fixtures/sample.pdf"
    raw = source.read_bytes()
    upload = tmp_path / "upload.pdf"
    upload.write_bytes(raw)
    store = MetadataStore("sqlite+aiosqlite:///" + str(tmp_path / "metadata.db"))
    await store.init()
    queue = IngestQueue()
    job_id = queue.enqueue(
        "sample.pdf",
        str(upload),
        ["engineering"],
        "test",
        category="uncategorized",
        auto_categorize=False,
        build_graph=False,
    )
    try:
        await queue._process_job(queue.get_job(job_id), MagicMock(), store)
        job = queue.get_job(job_id)
        assert job.step == IngestStep.COMPLETE and not upload.exists()
        doc = await store.get_document(job.doc_id)
        assert doc.content_hash == hashlib.sha256(raw).hexdigest()
        stream, _ = OriginalStore().open_verified(doc.doc_id, doc.content_hash)
        try:
            assert stream.read() == raw
        finally:
            stream.close()
    finally:
        await store.engine.dispose()
