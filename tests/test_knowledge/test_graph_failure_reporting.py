"""Graph failures must not be reported as successful empty extraction."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.knowledge import graph_rag
from src.ingestion.queue import IngestQueue, IngestStep


@pytest.fixture
def rag(monkeypatch):
    rag = SimpleNamespace(
        ainsert=AsyncMock(return_value="tracking-id"),
        doc_status=SimpleNamespace(
            get_docs_by_statuses=AsyncMock(return_value={}),
            get_by_id=AsyncMock(return_value={"status": "processed"}),
        ),
        chunk_entity_relation_graph=SimpleNamespace(
            get_all_nodes=AsyncMock(return_value=[]),
            get_all_edges=AsyncMock(return_value=[]),
            index_done_callback=AsyncMock(),
        ),
    )
    store = SimpleNamespace(engine=SimpleNamespace(dispose=AsyncMock()), init=AsyncMock(), list_documents=AsyncMock(return_value=[]))
    monkeypatch.setattr("src.db.metadata.MetadataStore", lambda: store)
    monkeypatch.setattr(graph_rag, "get_lightrag", AsyncMock(return_value=rag))
    monkeypatch.setattr(graph_rag, "release_pipeline_busy", AsyncMock())
    monkeypatch.setattr(graph_rag, "_invalidate_query_cache", lambda: None)
    monkeypatch.setattr(graph_rag, "_insert_lock", asyncio.Lock())
    return rag


@pytest.mark.asyncio
async def test_insert_exception_reaches_caller(rag):
    rag.ainsert.side_effect = RuntimeError("model server unavailable")
    with pytest.raises(RuntimeError, match="model server unavailable"):
        await graph_rag.insert_document("Router A connects to B", "doc1", "test.txt")
    graph_rag.release_pipeline_busy.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("record", [
    {"status": "failed", "error_msg": "invalid model output"},
    {"status": "pending"}, {"status": "processing"}, None,
])
async def test_returning_insert_is_not_proof_of_success(rag, record):
    rag.doc_status.get_by_id.return_value = record
    with pytest.raises(RuntimeError, match="LightRAG document status"):
        await graph_rag.insert_document("Router A connects to B", "doc1", "test.txt")


@pytest.mark.asyncio
async def test_processed_document_succeeds(rag):
    assert await graph_rag.insert_document("A connects to B", "doc1", "test.txt") == "tracking-id"
    rag.chunk_entity_relation_graph.index_done_callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_count_read_failure_is_not_zero(rag):
    rag.chunk_entity_relation_graph.get_all_nodes.side_effect = OSError("graph unreadable")
    with pytest.raises(OSError, match="graph unreadable"):
        await graph_rag.get_graph_counts()


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [None, "", " \n "])
async def test_empty_model_response_fails(monkeypatch, output):
    monkeypatch.setattr(graph_rag, "openai_complete_if_cache", AsyncMock(return_value=output))
    with pytest.raises(RuntimeError, match="model returned no text"):
        await graph_rag._llm_func("Extract the relationships")


@pytest.mark.asyncio
async def test_model_record_repair_is_preserved(monkeypatch):
    monkeypatch.setattr(graph_rag, "openai_complete_if_cache", AsyncMock(
        return_value="entity<|#|>Router A<|#|>An edge router"))
    assert await graph_rag._llm_func("Extract") == "entity<|#|>Router A<|#|>category<|#|>An edge router"


@pytest.fixture
def queue_job(monkeypatch):
    monkeypatch.setattr(graph_rag, "insert_document", AsyncMock())
    monkeypatch.setattr(graph_rag, "get_graph_counts", AsyncMock(return_value=(0, 0)))
    monkeypatch.setattr(graph_rag, "release_pipeline_busy", AsyncMock())
    monkeypatch.setattr(graph_rag.settings, "kg_extract_max_retries", 1)
    q = IngestQueue()
    job = q.get_job(q.enqueue("test.txt", "/tmp/test.txt", [], "test"))
    return q, job


@pytest.mark.asyncio
async def test_empty_graph_warns_without_destroying_document(queue_job):
    q, job = queue_job
    await q._build_knowledge_graph(job, "Router A connects to Router B", "doc1")
    q.complete_job(job.job_id, "doc1", 4)
    assert job.step == IngestStep.COMPLETE
    assert "graph is empty" in job.progress
    assert job.warnings == [job.progress]


@pytest.mark.asyncio
async def test_existing_entities_are_not_false_empty_graph(queue_job, monkeypatch):
    monkeypatch.setattr(graph_rag, "get_graph_counts", AsyncMock(return_value=(2, 1)))
    q, job = queue_job
    await q._build_knowledge_graph(job, "Router A connects to Router B", "doc1")
    assert "0 new entities" in job.progress
    assert not job.warnings


@pytest.mark.asyncio
async def test_empty_source_does_not_call_model(queue_job):
    q, job = queue_job
    await q._build_knowledge_graph(job, " \n ", "doc1")
    graph_rag.insert_document.assert_not_awaited()
    assert "no extracted text" in job.warnings[0]


@pytest.mark.asyncio
async def test_queue_keeps_graph_failure_visible_after_completion(queue_job):
    q, job = queue_job
    graph_rag.insert_document.side_effect = RuntimeError("invalid model output")
    await q._build_knowledge_graph(job, "Router A connects to Router B", "doc1")
    q.complete_job(job.job_id, "doc1", 4)
    assert job.step == IngestStep.COMPLETE
    assert "Knowledge graph failed: invalid model output" in job.progress
    assert job.warnings == [job.progress]


@pytest.mark.asyncio
async def test_timeout_and_unreadable_counts_do_not_fail_indexing(queue_job):
    q, job = queue_job
    graph_rag.insert_document.side_effect = asyncio.TimeoutError()
    graph_rag.get_graph_counts.side_effect = [(0, 0), OSError("graph unreadable")]
    await q._build_knowledge_graph(job, "Router A connects to Router B", "doc1")
    assert "timed out" in job.progress
    assert "partial counts unavailable" in job.progress
    assert job.warnings


@pytest.mark.asyncio
async def test_unreadable_graph_is_not_reported_as_success(queue_job):
    q, job = queue_job
    graph_rag.get_graph_counts.side_effect = OSError("graph unreadable")
    await q._build_knowledge_graph(job, "Router A connects to Router B", "doc1")
    assert "Knowledge graph failed: graph unreadable" in job.progress
    graph_rag.insert_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_persistence_failure_reaches_caller(rag):
    rag.chunk_entity_relation_graph.index_done_callback.side_effect = OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        await graph_rag.insert_document("A connects to B", "doc1", "test.txt")


@pytest.mark.asyncio
async def test_partial_graph_warning_shows_retained_counts(queue_job, monkeypatch):
    from src.knowledge.resilient_lightrag import PartialGraphError
    q, job = queue_job
    graph_rag.insert_document.side_effect = PartialGraphError('Knowledge graph incomplete: 1 of 3 chunks failed')
    monkeypatch.setattr(graph_rag, 'get_document_graph_counts', AsyncMock(return_value=(7, 4)))
    assert await q._build_knowledge_graph(job, 'Router A connects to B', 'doc1') is False
    assert job.entity_count == 7 and job.relationship_count == 4
    assert job.warnings == ['Knowledge graph incomplete: 1 of 3 chunks failed']
