# Async Query Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an async external query mode — caller `POST`s a question, gets a token immediately, then polls `GET …/{token}` for live step progress and the final answer.

**Architecture:** An in-memory, TTL-bounded job queue (`QueryJobQueue`) with a bounded pool of background asyncio workers, mirroring the existing `src/ingestion/queue.py`. Each worker runs the agent pipeline through a shared streaming helper that reports per-node progress via a callback while preserving the existing cache-parity path. Two new endpoints on the existing `/api/v1` query router; results are owner-scoped.

**Tech Stack:** FastAPI, Pydantic, LangGraph (`astream`), asyncio, pytest (`asyncio_mode = "auto"`).

**Reference spec:** `docs/superpowers/specs/2026-05-31-async-query-mode-design.md`

---

## File Structure

- **Create** `src/api/query_jobs.py` — `QueryStatus`, `QueryJob`, `QueryJobQueue`, `STEP_LABELS`, singleton `query_queue`. Owns all async-query job state + the worker pool.
- **Modify** `src/agent/graph.py` — add `run_agent_streamed(..., step_callback=None)`: single-pass graph stream that fires `step_callback(node_name)` per node and returns a `RAGResponse`.
- **Modify** `src/generation/rag_chain.py` — add `agent_query_streamed(..., step_callback=None)` (cache lookup → stream → cache_store); make `agent_query` delegate to it with `step_callback=None` (no behavior change).
- **Modify** `src/api/models.py` — add `AsyncQuerySubmitResponse`, `AsyncQueryStatusResponse`.
- **Modify** `src/api/routes_query.py` — add `POST /api/v1/query/async` and `GET /api/v1/query/async/{token}`.
- **Modify** `src/config.py` — add `max_parallel_async_query`, `async_query_ttl_seconds`.
- **Create** `tests/test_api/test_query_jobs.py` — queue unit tests.
- **Create** `tests/test_api/test_routes_query_async.py` — endpoint tests.
- **Create** `tests/test_generation/test_agent_query_streamed.py` — shared-helper tests.

No change to `src/main.py` — the new endpoints attach to the existing `query_router`.

---

## Task 1: Config settings

**Files:**
- Modify: `src/config.py:54`

- [ ] **Step 1: Add the two settings**

In `src/config.py`, immediately after the line `max_parallel_ingestion: int = 3  # concurrent file ingestion jobs` (line 54), add:

```python
    max_parallel_async_query: int = 3  # concurrent async query worker slots
    async_query_ttl_seconds: int = 3600  # how long finished async jobs are retained
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from src.config import settings; print(settings.max_parallel_async_query, settings.async_query_ttl_seconds)"`
Expected: `3 3600`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: config for async query pool size + result TTL"
```

---

## Task 2: Shared streaming helper in the agent graph

Add a single-pass streaming runner so progress can be observed without running the graph twice (`run_agent_with_trace` currently streams *and* re-invokes — do not copy that; accumulate the final state from the stream).

**Files:**
- Modify: `src/agent/graph.py` (add after `run_agent`, near line 309)
- Test: `tests/test_generation/test_agent_query_streamed.py` (helper-level test added in Task 3; graph-level behavior is covered there via the rag_chain wrapper)

- [ ] **Step 1: Add `run_agent_streamed`**

In `src/agent/graph.py`, directly after the `run_agent` function (after line 308), add:

```python
async def run_agent_streamed(
    question: str,
    user_groups: list[str],
    vector_store: VectorStore,
    schema_registry: SchemaRegistry,
    metadata_store: MetadataStore | None = None,
    step_callback=None,
) -> RAGResponse:
    """Run the agent graph once, emitting step_callback(node_name) per node.

    Single-pass: the final state is accumulated from the stream itself, so the
    graph executes exactly once (unlike run_agent_with_trace).
    """
    graph = create_agent_graph(vector_store=vector_store, schema_registry=schema_registry, metadata_store=metadata_store)
    initial_state = AgentState(
        question=question, original_question=question, user_groups=user_groups,
        query_type=None, sub_tasks=[], retrieved_chunks=[], sql_results=[],
        retrieval_attempts=0, needs_reretrieval=False, reformulated_query="",
        answer="", citations=[], warnings=[],
    )
    final_state = dict(initial_state)
    async for event in graph.astream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if isinstance(node_output, dict):
                final_state.update(node_output)
            if step_callback is not None:
                step_callback(node_name)
    return RAGResponse(
        answer=final_state.get("answer", "I could not find any relevant information."),
        citations=final_state.get("citations", []),
    )
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from src.agent.graph import run_agent_streamed; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/agent/graph.py
git commit -m "feat: single-pass run_agent_streamed with per-node step callback"
```

---

## Task 3: Cache-aware streaming wrapper in rag_chain

`agent_query_streamed` adds cache parity (same `judged_cache_lookup` / `cache_store` path as `agent_query`) around `run_agent_streamed`. `agent_query` becomes a thin delegate so the existing blocking endpoint is unchanged.

**Files:**
- Modify: `src/generation/rag_chain.py:64-104`
- Test: `tests/test_generation/test_agent_query_streamed.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generation/test_agent_query_streamed.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch
from src.generation.rag_chain import agent_query_streamed, agent_query, RAGResponse
from src.retrieval.models import Citation


async def test_streamed_fires_callback_per_node_on_cache_miss():
    seen = []
    miss = MagicMock(accepted=False, query_vector=None)
    resp = RAGResponse(answer="A", citations=[])

    async def fake_run_streamed(*args, **kwargs):
        cb = kwargs["step_callback"]
        for node in ("classify", "retrieve", "synthesize"):
            cb(node)
        return resp

    with patch("src.generation.rag_chain.judged_cache_lookup", new_callable=AsyncMock, return_value=miss):
        with patch("src.agent.graph.run_agent_streamed", side_effect=fake_run_streamed):
            out = await agent_query_streamed(
                "q", ["finance"], MagicMock(), MagicMock(), None,
                step_callback=seen.append,
            )
    assert out.answer == "A"
    assert seen == ["classify", "retrieve", "synthesize"]


async def test_streamed_returns_cached_without_running_graph():
    cached = {"answer": "cached!", "citations": [], "cached_query": "old q"}
    decision = MagicMock(accepted=True, cached=cached, query_vector=None)
    with patch("src.generation.rag_chain.judged_cache_lookup", new_callable=AsyncMock, return_value=decision):
        with patch("src.agent.graph.run_agent_streamed", new_callable=AsyncMock) as run:
            out = await agent_query_streamed("q", ["finance"], MagicMock(), MagicMock(), None)
    assert out.cached is True
    assert out.answer == "cached!"
    run.assert_not_awaited()


async def test_agent_query_delegates_with_no_callback():
    miss = MagicMock(accepted=False, query_vector=None)
    resp = RAGResponse(answer="Z", citations=[])
    with patch("src.generation.rag_chain.judged_cache_lookup", new_callable=AsyncMock, return_value=miss):
        with patch("src.agent.graph.run_agent_streamed", new_callable=AsyncMock, return_value=resp) as run:
            out = await agent_query("q", ["finance"], MagicMock(), MagicMock(), None)
    assert out.answer == "Z"
    assert run.await_args.kwargs["step_callback"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generation/test_agent_query_streamed.py -v`
Expected: FAIL — `ImportError: cannot import name 'agent_query_streamed'`

- [ ] **Step 3: Implement the wrapper and refactor `agent_query`**

In `src/generation/rag_chain.py`, replace the entire `agent_query` function (lines 64-104) with:

```python
async def agent_query_streamed(
    question: str, user_groups: list[str], vector_store, schema_registry,
    metadata_store=None, step_callback=None,
) -> RAGResponse:
    # Shared cache decision (embed -> lookup -> LLM applicability judge) — same
    # path the admin playground uses, so the two cannot diverge.
    decision = await judged_cache_lookup(question, user_groups)
    if decision.accepted:
        cached = decision.cached
        citations = [
            Citation(
                doc_id=c.get("doc_id", ""), filename=c.get("filename", ""),
                doc_type=c.get("doc_type", ""), chunk_index=c.get("chunk_index", 0),
                page=c.get("page"), snippet=c.get("snippet", ""),
                relevance=c.get("relevance", 0.0),
            )
            for c in cached.get("citations", [])
        ]
        return RAGResponse(answer=cached["answer"], citations=citations,
                           cached=True, cached_query=cached.get("cached_query"))

    # run_agent_streamed is imported lazily to avoid a circular import: src.agent.graph
    # imports RAGResponse from this module. Tests patch src.agent.graph.run_agent_streamed.
    from src.agent.graph import run_agent_streamed
    result = await run_agent_streamed(
        question=question, user_groups=user_groups, vector_store=vector_store,
        schema_registry=schema_registry, metadata_store=metadata_store,
        step_callback=step_callback,
    )

    if decision.query_vector is not None:
        try:
            citation_dicts = [
                {"doc_id": c.doc_id, "filename": c.filename, "doc_type": c.doc_type,
                 "chunk_index": c.chunk_index, "page": c.page, "snippet": c.snippet,
                 "relevance": c.relevance}
                for c in result.citations
            ]
            source_ids = list({c.doc_id for c in result.citations})
            cache_store(
                query_text=question, query_vector=decision.query_vector,
                answer=result.answer, citations=citation_dicts,
                user_groups=user_groups, source_doc_ids=source_ids,
            )
        except Exception:
            pass

    return result


async def agent_query(question: str, user_groups: list[str], vector_store, schema_registry, metadata_store=None) -> RAGResponse:
    return await agent_query_streamed(
        question=question, user_groups=user_groups, vector_store=vector_store,
        schema_registry=schema_registry, metadata_store=metadata_store,
        step_callback=None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generation/test_agent_query_streamed.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the existing query-path tests for regression**

Run: `pytest tests/test_api/test_routes_query.py -v`
Expected: PASS (existing tests still green — `agent_query` behaves identically)

- [ ] **Step 6: Commit**

```bash
git add src/generation/rag_chain.py tests/test_generation/test_agent_query_streamed.py
git commit -m "feat: agent_query_streamed (cache parity + step callback); agent_query delegates"
```

---

## Task 4: The job queue (state + token issuance)

Build `QueryJobQueue` state management first (no worker yet): enqueue/get/update/complete/fail + TTL eviction + step labels.

**Files:**
- Create: `src/api/query_jobs.py`
- Test: `tests/test_api/test_query_jobs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api/test_query_jobs.py`:

```python
import asyncio
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api/test_query_jobs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.api.query_jobs'`

- [ ] **Step 3: Implement state + TTL (no worker yet)**

Create `src/api/query_jobs.py`:

```python
from __future__ import annotations
"""In-memory, TTL-bounded job queue for async external queries.

Mirrors src/ingestion/queue.py (bounded worker pool, step tracking) and
src/mcp/jobs.py (TTL eviction). In-memory only: a restart drops jobs and the
caller must resubmit (see the design spec).
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)

# Agent-graph node names -> human-readable step labels shown to API callers.
STEP_LABELS = {
    "queued": "queued",
    "cache_check": "checking cache",
    "classify": "classifying question",
    "retrieve": "retrieving documents",
    "enrich": "searching knowledge graph",
    "merge": "merging results",
    "synthesize": "synthesizing answer",
    "complete": "complete",
}


class QueryStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class QueryJob:
    token: str
    question: str
    username: str
    groups: list[str]
    status: QueryStatus = QueryStatus.QUEUED
    step: str = "queued"
    answer: str | None = None
    citations: list[dict] = field(default_factory=list)
    cached: bool = False
    cached_query: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


_TERMINAL = (QueryStatus.COMPLETE, QueryStatus.FAILED)


class QueryJobQueue:
    def __init__(self, ttl_seconds: int | None = None):
        if ttl_seconds is None:
            from src.config import settings
            ttl_seconds = settings.async_query_ttl_seconds
        self._ttl = ttl_seconds
        self._jobs: dict[str, QueryJob] = {}
        self._queue: asyncio.Queue | None = None
        self._worker_running = False
        self._stores: tuple | None = None  # (vector_store, schema_registry, metadata_store)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            t for t, j in self._jobs.items()
            if j.status in _TERMINAL and j.completed_at is not None
            and now - j.completed_at > self._ttl
        ]
        for t in expired:
            del self._jobs[t]

    def enqueue(self, question: str, username: str, groups: list[str]) -> str:
        self._evict_expired()
        token = str(uuid.uuid4())
        self._jobs[token] = QueryJob(token=token, question=question, username=username, groups=groups)
        if self._queue is not None:
            self._queue.put_nowait(token)
        return token

    def get_job(self, token: str) -> QueryJob | None:
        self._evict_expired()
        return self._jobs.get(token)

    def update_step(self, token: str, node_name: str) -> None:
        job = self._jobs.get(token)
        if job:
            job.status = QueryStatus.PROCESSING
            job.step = STEP_LABELS.get(node_name, node_name)

    def complete(self, token: str, answer: str, citations: list[dict], cached: bool, cached_query: str | None) -> None:
        job = self._jobs.get(token)
        if job:
            job.status = QueryStatus.COMPLETE
            job.step = STEP_LABELS["complete"]
            job.answer = answer
            job.citations = citations
            job.cached = cached
            job.cached_query = cached_query
            job.completed_at = time.time()

    def fail(self, token: str, error: str) -> None:
        job = self._jobs.get(token)
        if job:
            job.status = QueryStatus.FAILED
            job.error = error
            job.completed_at = time.time()


# Singleton
query_queue = QueryJobQueue()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api/test_query_jobs.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/api/query_jobs.py tests/test_api/test_query_jobs.py
git commit -m "feat: QueryJobQueue state + TTL eviction + step labels"
```

---

## Task 5: The worker pool

Add the bounded background worker that drains the queue and runs the pipeline via `agent_query_streamed`, reporting progress through the step callback.

**Files:**
- Modify: `src/api/query_jobs.py`
- Test: `tests/test_api/test_query_jobs.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api/test_query_jobs.py`:

```python
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
    assert "kaboom" in job.error


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api/test_query_jobs.py -k worker -v`
Expected: FAIL — `AttributeError: 'QueryJobQueue' object has no attribute 'start_worker'`

- [ ] **Step 3: Implement the worker pool**

In `src/api/query_jobs.py`, add this import near the top (after `from enum import StrEnum`):

```python
from src.generation.rag_chain import agent_query_streamed
```

Then add a class attribute and the worker methods to `QueryJobQueue` (insert immediately before the `# Singleton` comment / after `fail`):

```python
    max_parallel: int = 3  # default; overridden from settings in start_worker

    async def start_worker(self, vector_store, schema_registry, metadata_store) -> None:
        """Start the bounded worker pool (idempotent)."""
        if self._worker_running:
            return
        self._stores = (vector_store, schema_registry, metadata_store)
        self._queue = asyncio.Queue()
        self._worker_running = True
        # Re-queue jobs enqueued before the worker started.
        for token, job in self._jobs.items():
            if job.status == QueryStatus.QUEUED:
                self._queue.put_nowait(token)
        from src.config import settings
        self.max_parallel = settings.max_parallel_async_query
        for _ in range(self.max_parallel):
            asyncio.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        import traceback
        while True:
            token = await self._queue.get()
            job = self._jobs.get(token)
            if job is None:
                self._queue.task_done()
                continue
            try:
                vector_store, schema_registry, metadata_store = self._stores
                job.status = QueryStatus.PROCESSING
                result = await agent_query_streamed(
                    question=job.question, user_groups=job.groups,
                    vector_store=vector_store, schema_registry=schema_registry,
                    metadata_store=metadata_store,
                    step_callback=lambda node, _t=token: self.update_step(_t, node),
                )
                citation_dicts = [
                    {"doc_id": c.doc_id, "filename": c.filename, "doc_type": c.doc_type,
                     "chunk_index": c.chunk_index, "page": c.page, "snippet": c.snippet,
                     "relevance": c.relevance}
                    for c in result.citations
                ]
                self.complete(token, answer=result.answer, citations=citation_dicts,
                              cached=result.cached, cached_query=result.cached_query)
            except Exception as e:
                logger.error(f"Async query {token} failed: {e}\n{traceback.format_exc()}")
                self.fail(token, str(e))
            self._queue.task_done()
```

Note: `result.citations` are `Citation` dataclasses; they are flattened to plain dicts here so the job holds JSON-serializable data for the status endpoint.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api/test_query_jobs.py -v`
Expected: PASS (10 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/api/query_jobs.py tests/test_api/test_query_jobs.py
git commit -m "feat: bounded async query worker pool (progress + complete/fail)"
```

---

## Task 6: Response models

**Files:**
- Modify: `src/api/models.py:44` (append after `QueryResponse`)

- [ ] **Step 1: Add the models**

At the end of `src/api/models.py`, append:

```python
class AsyncQuerySubmitResponse(BaseModel):
    token: str
    status: str

class AsyncQueryStatusResponse(BaseModel):
    token: str
    status: str
    step: str
    answer: str | None = None
    citations: list[CitationResponse] = []
    cached: bool = False
    cached_query: str | None = None
    error: str | None = None
    created_at: float
    completed_at: float | None = None
```

- [ ] **Step 2: Verify import**

Run: `python -c "from src.api.models import AsyncQuerySubmitResponse, AsyncQueryStatusResponse; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/api/models.py
git commit -m "feat: async query submit/status response models"
```

---

## Task 7: Endpoints (submit + poll)

**Files:**
- Modify: `src/api/routes_query.py`
- Test: `tests/test_api/test_routes_query_async.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api/test_routes_query_async.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from src.main import create_app
from src.auth.jwt import create_token
from src.api.query_jobs import query_queue, QueryStatus, QueryJob


@pytest.fixture
def auth_headers():
    token = create_token(username="mike", groups=["finance"])
    return {"Authorization": f"Bearer {token}", "X-API-Key": "test-key-1"}


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _clear_queue():
    query_queue._jobs.clear()
    yield
    query_queue._jobs.clear()


def test_submit_returns_token_and_queued(client, auth_headers):
    with patch("src.api.routes_query.query_queue.start_worker", new_callable=AsyncMock):
        with patch("src.api.routes_query.get_vector_store", return_value=MagicMock()):
            with patch("src.api.routes_query.get_schema_registry", return_value=MagicMock()):
                with patch("src.api.routes_query.get_metadata_store", return_value=MagicMock()):
                    resp = client.post("/api/v1/query/async", json={"question": "slow q?"}, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"
    assert data["token"]
    job = query_queue.get_job(data["token"])
    assert job.username == "mike"
    assert job.question == "slow q?"


def test_submit_requires_auth(client):
    resp = client.post("/api/v1/query/async", json={"question": "x"})
    assert resp.status_code in (401, 403)


def test_poll_returns_completed_answer(client, auth_headers):
    query_queue._jobs["tok-1"] = QueryJob(
        token="tok-1", question="q", username="mike", groups=["finance"],
        status=QueryStatus.COMPLETE, step="complete", answer="done!",
        citations=[{"doc_id": "d1", "filename": "p.pdf", "doc_type": "pdf",
                    "chunk_index": 0, "page": 3, "snippet": "s", "relevance": 0.9}],
        completed_at=123.0,
    )
    resp = client.get("/api/v1/query/async/tok-1", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "complete"
    assert data["answer"] == "done!"
    assert data["citations"][0]["filename"] == "p.pdf"


def test_poll_unknown_token_404(client, auth_headers):
    resp = client.get("/api/v1/query/async/missing", headers=auth_headers)
    assert resp.status_code == 404


def test_poll_other_users_token_404(client, auth_headers):
    query_queue._jobs["tok-2"] = QueryJob(
        token="tok-2", question="q", username="someone-else", groups=["hr"],
        status=QueryStatus.PROCESSING, step="retrieving documents",
    )
    resp = client.get("/api/v1/query/async/tok-2", headers=auth_headers)
    assert resp.status_code == 404


def test_poll_requires_auth(client):
    resp = client.get("/api/v1/query/async/whatever")
    assert resp.status_code in (401, 403)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api/test_routes_query_async.py -v`
Expected: FAIL — 404s on the new routes (routes not defined yet)

- [ ] **Step 3: Implement the endpoints**

In `src/api/routes_query.py`, update the imports and add `get_metadata_store` + the two routes. Replace the import block (lines 1-6) with:

```python
from fastapi import APIRouter, Depends, HTTPException
from src.api.models import (
    CitationResponse, QueryRequest, QueryResponse,
    AsyncQuerySubmitResponse, AsyncQueryStatusResponse,
)
from src.api.routes_ingest import get_vector_store, get_schema_registry, get_metadata_store
from src.auth.dependencies import require_auth
from src.auth.models import UserContext
from src.generation.rag_chain import agent_query
from src.api.query_jobs import query_queue
```

Then append, after the existing `query` function (after line 22):

```python
@router.post("/query/async", response_model=AsyncQuerySubmitResponse)
async def query_async(request: QueryRequest, user: UserContext = Depends(require_auth)):
    """Submit a question for async processing. Returns a token to poll for status/result."""
    await query_queue.start_worker(get_vector_store(), get_schema_registry(), get_metadata_store())
    token = query_queue.enqueue(question=request.question, username=user.username, groups=user.groups)
    return AsyncQuerySubmitResponse(token=token, status="queued")


@router.get("/query/async/{token}", response_model=AsyncQueryStatusResponse)
async def query_async_status(token: str, user: UserContext = Depends(require_auth)):
    """Poll an async query by token. Owner-scoped: another user's token returns 404."""
    job = query_queue.get_job(token)
    if job is None or job.username != user.username:
        raise HTTPException(status_code=404, detail="Job not found")
    return AsyncQueryStatusResponse(
        token=job.token,
        status=str(job.status),
        step=job.step,
        answer=job.answer,
        citations=[CitationResponse(**c) for c in job.citations],
        cached=job.cached,
        cached_query=job.cached_query,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api/test_routes_query_async.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/api/routes_query.py tests/test_api/test_routes_query_async.py
git commit -m "feat: POST /query/async + GET /query/async/{token} (owner-scoped)"
```

---

## Task 8: Full-suite regression + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the affected test areas**

Run: `pytest tests/test_api tests/test_generation -v`
Expected: PASS (existing + new tests green)

- [ ] **Step 2: Run the full suite to catch unexpected breakage**

Run: `pytest -q`
Expected: No new failures versus the pre-change baseline. (Per project memory, 5 pre-existing `test_routes.py` admin-auth failures may already be red on master — confirm they are the *same* failures, not new ones.)

- [ ] **Step 3: Document the manual smoke test in the PR**

Record these commands (run against a deployed instance with a real token) for the PR description — not executed in CI:

```bash
# 1. Get a JWT
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"username":"mike","password":"x","groups":["executives"]}' | jq -r .access_token)

# 2. Submit async
TOK=$(curl -s -X POST localhost:8000/api/v1/query/async \
  -H "Authorization: Bearer $TOKEN" -H 'X-API-Key: dev-key-1' \
  -H 'Content-Type: application/json' \
  -d '{"question":"what is the pay rate for florida?"}' | jq -r .token)

# 3. Poll until complete
curl -s localhost:8000/api/v1/query/async/$TOK \
  -H "Authorization: Bearer $TOKEN" -H 'X-API-Key: dev-key-1' | jq
```

Expected: poll shows `status` moving `queued`→`processing` (with `step` like "retrieving documents") then `complete` with `answer` + `citations`.

- [ ] **Step 4: Final commit (if any doc tweaks)**

```bash
git add -A
git commit -m "test: async query mode regression verification notes" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- In-memory + TTL → Task 4 (`_evict_expired`, `ttl_seconds`). ✓
- Rich step progress → Task 2/3 (`step_callback`), Task 4 (`STEP_LABELS`, `update_step`). ✓
- Owner-scoped → Task 7 (`job.username != user.username → 404`). ✓
- Bounded pool → Task 5 (`max_parallel`, N workers). ✓
- Submit endpoint + poll endpoint → Task 7. ✓
- Response models → Task 6. ✓
- Config → Task 1. ✓
- Cache parity preserved → Task 3 (`agent_query` delegates). ✓
- Tests (queue/endpoint/helper) → Tasks 3,4,5,7. ✓

**Type consistency:** `agent_query_streamed(step_callback=...)` defined in Task 3, called in Task 5. `QueryJobQueue.start_worker(vector_store, schema_registry, metadata_store)` defined Task 5, called Task 7. `QueryJob` fields used in Task 7 (`username`, `status`, `step`, `answer`, `citations`, `cached`, `cached_query`, `error`, `created_at`, `completed_at`) all defined Task 4. `complete(token, answer, citations, cached, cached_query)` signature consistent across Tasks 4/5. `CitationResponse(**c)` requires `c` dicts to carry exactly the CitationResponse fields — the worker (Task 5) and the test seed (Task 7) both produce those keys. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. ✓
