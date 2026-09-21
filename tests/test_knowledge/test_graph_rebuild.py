"""Recovery must use all retained passages and never roll back the existing index."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.ingestion.queue import IngestQueue, IngestStep
from src.knowledge.rebuild import read_indexed_graph_text
from src.knowledge import graph_rag
from src.retrieval.models import ChunkMetadata, RetrievedChunk


def chunk(index, text):
    return RetrievedChunk(text=text, score=0, metadata=ChunkMetadata(
        doc_id='d', filename='guide.pdf', doc_type='pdf', chunk_index=index,
        start_char=index * 100, acl_groups=['private']))


def test_recovery_reads_beyond_200_chunks_and_strips_synthetic_summary():
    chunks = [chunk(i, f'Document: guide.pdf\nSummary: Invented summary\n\nPassage {i}') for i in range(251)]
    def page(doc_id, groups, offset, limit):
        assert groups == ['ALL']
        return chunks[offset:offset + limit], offset + limit < len(chunks)
    vectors = SimpleNamespace(read_document_page=Mock(side_effect=page))
    text = read_indexed_graph_text(vectors, 'd')
    assert 'Passage 250' in text and 'Invented summary' not in text
    assert vectors.read_document_page.call_count == 3


def test_missing_index_read_fails_instead_of_appearing_empty():
    vectors = SimpleNamespace(read_document_page=Mock(side_effect=OSError('disk failure')))
    with pytest.raises(OSError):
        read_indexed_graph_text(vectors, 'd')


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [False, True])
async def test_rebuild_only_updates_graph_and_reports_failure(monkeypatch, failure):
    q = IngestQueue()
    doc = SimpleNamespace(doc_id='d', filename='guide.pdf', doc_type='pdf',
                          acl_groups=['private'], dataset_id=5, chunk_count=251)
    metadata = SimpleNamespace(get_document=AsyncMock(return_value=doc), delete_document=AsyncMock())
    vectors = SimpleNamespace(read_document_page=Mock(return_value=([chunk(0, 'Router A connects to Router B')], False)),
                              delete_by_doc_id=Mock(), upsert=Mock())
    insert = AsyncMock(side_effect=RuntimeError('TLS failed') if failure else None)
    monkeypatch.setattr(graph_rag, 'insert_document', insert)
    monkeypatch.setattr(graph_rag, 'get_graph_counts', AsyncMock(return_value=(2, 1)))
    monkeypatch.setattr(graph_rag, 'get_document_graph_counts', AsyncMock(return_value=(2, 1)))
    monkeypatch.setattr(graph_rag, 'release_pipeline_busy', AsyncMock())
    monkeypatch.setattr(graph_rag.settings, 'kg_extract_max_retries', 1)
    job = q.get_job(q.enqueue_graph_rebuild(doc))
    assert q.enqueue_graph_rebuild(doc) == job.job_id
    await q._process_job(job, vectors, metadata)
    assert job.step == (IngestStep.FAILED if failure else IngestStep.COMPLETE)
    insert.assert_awaited_once_with('Router A connects to Router B', doc_id='d', filename='guide.pdf', rebuild=True)
    # This is also called by the worker on an exception/shutdown. It must not
    # remove the existing source's metadata, vectors, or structured tables.
    await q._cleanup_failed_job(job, vectors, metadata)
    vectors.delete_by_doc_id.assert_not_called()
    vectors.upsert.assert_not_called()
    metadata.delete_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleted_document_is_not_rebuilt():
    q = IngestQueue()
    doc = SimpleNamespace(doc_id='gone', filename='guide.pdf', acl_groups=[], dataset_id=0, chunk_count=1)
    job = q.get_job(q.enqueue_graph_rebuild(doc))
    with pytest.raises(RuntimeError, match='no longer exists'):
        await q._process_job(job, Mock(), SimpleNamespace(get_document=AsyncMock(return_value=None)))


@pytest.mark.asyncio
async def test_rebuilds_wait_before_starting_their_processing_budget():
    q = IngestQueue()
    active = 0
    maximum = 0
    async def rebuild(*args):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(.01)
        active -= 1
    q._rebuild_graph_job = rebuild
    doc = SimpleNamespace(doc_id='d', filename='guide.pdf', acl_groups=[], dataset_id=0, chunk_count=1)
    first=q.get_job(q.enqueue_graph_rebuild(doc))
    doc.doc_id='e'
    second=q.get_job(q.enqueue_graph_rebuild(doc))
    await asyncio.gather(q._process_job(first, None, None), q._process_job(second, None, None))
    assert maximum == 1
