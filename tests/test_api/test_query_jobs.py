import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.api.query_jobs import QueryJobQueue, QueryStatus, STEP_LABELS


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
