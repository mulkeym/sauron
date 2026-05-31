import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.api.query_jobs import QueryJobQueue, QueryStatus, STEP_LABELS, QueueFullError


def test_enqueue_returns_token_and_queued_job():
    q = QueryJobQueue()
    token = q.enqueue(question="what is the pay rate?", username="mike", groups=["finance"])
    assert token
    job = q.get_job(token)
    assert job.question == "what is the pay rate?"
    assert job.username == "mike"
    assert job.status == QueryStatus.QUEUED
    assert job.step == "queued"


def test_get_nonexistent_token():
    q = QueryJobQueue()
    assert q.get_job("nope") is None


def test_update_step_maps_node_to_label():
    q = QueryJobQueue()
    token = q.enqueue(question="x", username="m", groups=[])
    q.update_step(token, "retrieve")
    assert q.get_job(token).status == QueryStatus.PROCESSING
    assert q.get_job(token).step == STEP_LABELS["retrieve"]


def test_complete_sets_answer_and_status():
    q = QueryJobQueue()
    token = q.enqueue(question="x", username="m", groups=[])
    q.complete(token, answer="hi", citations=[{"filename": "a.pdf"}], cached=False, cached_query=None)
    job = q.get_job(token)
    assert job.status == QueryStatus.COMPLETE
    assert job.answer == "hi"
    assert job.citations == [{"filename": "a.pdf"}]
    assert job.completed_at > 0


def test_fail_sets_error_and_status():
    q = QueryJobQueue()
    token = q.enqueue(question="x", username="m", groups=[])
    q.fail(token, "boom")
    job = q.get_job(token)
    assert job.status == QueryStatus.FAILED
    assert job.error == "boom"


def test_ttl_evicts_finished_jobs_lazily():
    q = QueryJobQueue(ttl_seconds=0)
    token = q.enqueue(question="x", username="m", groups=[])
    q.complete(token, answer="done", citations=[], cached=False, cached_query=None)
    # ttl=0 means an already-completed job is expired on next access
    assert q.get_job(token) is None


def test_ttl_does_not_evict_running_jobs():
    q = QueryJobQueue(ttl_seconds=0)
    token = q.enqueue(question="x", username="m", groups=[])
    q.update_step(token, "classify")  # PROCESSING, not terminal
    assert q.get_job(token) is not None


@pytest.mark.asyncio
async def test_worker_processes_job_to_complete():
    q = QueryJobQueue()
    from src.generation.rag_chain import RAGResponse
    resp = RAGResponse(answer="42", citations=[], cached=False, cached_query=None)

    async def fake_streamed(*args, **kwargs):
        kwargs["step_callback"]("classify")
        return resp

    with patch("src.api.query_jobs.agent_query_streamed", side_effect=fake_streamed):
        await q.start_worker(MagicMock(), MagicMock(), MagicMock())
        token = q.enqueue(question="q", username="m", groups=[])
        for _ in range(50):
            if q.get_job(token).status == QueryStatus.COMPLETE:
                break
            await asyncio.sleep(0.02)
    job = q.get_job(token)
    assert job.status == QueryStatus.COMPLETE
    assert job.answer == "42"


@pytest.mark.asyncio
async def test_worker_marks_failed_on_exception():
    q = QueryJobQueue()

    async def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    with patch("src.api.query_jobs.agent_query_streamed", side_effect=boom):
        await q.start_worker(MagicMock(), MagicMock(), MagicMock())
        token = q.enqueue(question="q", username="m", groups=[])
        for _ in range(50):
            if q.get_job(token).status == QueryStatus.FAILED:
                break
            await asyncio.sleep(0.02)
    job = q.get_job(token)
    assert job.status == QueryStatus.FAILED
    # Raw exception text is logged server-side, not returned to the caller.
    assert job.error == "Query processing failed"


@pytest.mark.asyncio
async def test_worker_pool_respects_max_parallel():
    q = QueryJobQueue()
    q.max_parallel = 2
    running = 0
    peak = 0

    async def slow(*args, **kwargs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        running -= 1
        from src.generation.rag_chain import RAGResponse
        return RAGResponse(answer="x", citations=[])

    with patch("src.api.query_jobs.agent_query_streamed", side_effect=slow):
        await q.start_worker(MagicMock(), MagicMock(), MagicMock())
        tokens = [q.enqueue(question=f"q{i}", username="m", groups=[]) for i in range(4)]
        for _ in range(100):
            if all(q.get_job(t).status == QueryStatus.COMPLETE for t in tokens):
                break
            await asyncio.sleep(0.02)
    assert peak <= 2
    assert all(q.get_job(t).status == QueryStatus.COMPLETE for t in tokens)


def test_enqueue_rejects_when_full():
    q = QueryJobQueue(max_jobs=2)
    q.enqueue(question="a", username="m", groups=[])
    q.enqueue(question="b", username="m", groups=[])
    with pytest.raises(QueueFullError):
        q.enqueue(question="c", username="m", groups=[])


@pytest.mark.asyncio
async def test_worker_times_out_long_job():
    q = QueryJobQueue(job_timeout=0.05)

    async def slow(*args, **kwargs):
        await asyncio.sleep(1.0)
        from src.generation.rag_chain import RAGResponse
        return RAGResponse(answer="late", citations=[])

    with patch("src.api.query_jobs.agent_query_streamed", side_effect=slow):
        await q.start_worker(MagicMock(), MagicMock(), MagicMock())
        token = q.enqueue(question="q", username="m", groups=[])
        for _ in range(50):
            if q.get_job(token).status == QueryStatus.FAILED:
                break
            await asyncio.sleep(0.02)
    job = q.get_job(token)
    assert job.status == QueryStatus.FAILED
    assert job.error == "Query timed out"
