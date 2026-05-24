"""Tests for IngestQueue._cleanup_failed_job — rolls back partial writes so a
failed/interrupted ingestion can't leave orphaned LanceDB chunks."""
import pytest

from src.ingestion.queue import IngestJob, IngestQueue, IngestStep


class FakeVectorStore:
    def __init__(self):
        self.deleted = []

    def delete_by_doc_id(self, doc_id):
        self.deleted.append(doc_id)


class FakeMetadataStore:
    def __init__(self):
        self.deleted = []

    async def delete_document(self, doc_id):
        self.deleted.append(doc_id)


def _job(doc_id="") -> IngestJob:
    j = IngestJob(
        job_id="j1", filename="f.xls", file_path="/tmp/f.xls",
        acl_groups=[], uploaded_by="test", step=IngestStep.EMBEDDING,
    )
    j.doc_id = doc_id
    return j


@pytest.mark.asyncio
async def test_cleanup_deletes_both_stores_by_doc_id():
    q = IngestQueue()
    vs, ms = FakeVectorStore(), FakeMetadataStore()
    await q._cleanup_failed_job(_job("doc-123"), vs, ms)
    assert vs.deleted == ["doc-123"]
    assert ms.deleted == ["doc-123"]


@pytest.mark.asyncio
async def test_cleanup_noop_without_doc_id():
    q = IngestQueue()
    vs, ms = FakeVectorStore(), FakeMetadataStore()
    await q._cleanup_failed_job(_job(""), vs, ms)
    assert vs.deleted == []
    assert ms.deleted == []


@pytest.mark.asyncio
async def test_cleanup_continues_if_vector_delete_raises():
    q = IngestQueue()
    ms = FakeMetadataStore()

    class Boom:
        def delete_by_doc_id(self, doc_id):
            raise RuntimeError("lancedb down")

    # Vector failure must not prevent metadata cleanup (or propagate)
    await q._cleanup_failed_job(_job("doc-9"), Boom(), ms)
    assert ms.deleted == ["doc-9"]
