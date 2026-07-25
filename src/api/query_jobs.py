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

from src.generation.rag_chain import agent_query_streamed

logger = logging.getLogger(__name__)

# Agent-graph node names -> human-readable step labels shown to API callers.
STEP_LABELS = {
    "queued": "queued",
    "cache_check": "checking cache",
    "classify": "classifying question",
    "classify.hints": "reading available data tables",
    "classify.llm": "classifying question",
    "classify.strategy": "checking strategy memory",
    "classify.done": "classification complete",
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
    skip_cache: bool = False
    status: QueryStatus = QueryStatus.QUEUED
    step: str = "queued"
    answer: str | None = None
    citations: list[dict] = field(default_factory=list)
    cached: bool = False
    cached_query: str | None = None
    error: str | None = None
    steps: list = field(default_factory=list)      # timeline: [{"step": label, "at": elapsed_s}, ...]
    classification: dict | None = None             # classify node detail: query_type, reason, sub_tasks, strategy_memory
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


_TERMINAL = (QueryStatus.COMPLETE, QueryStatus.FAILED)

# Generic, caller-facing message. The full exception is logged server-side; we do
# not return raw exception text to external callers (the sync /query path returns
# a generic 500 with no body, so the async path must not leak more than that).
_GENERIC_ERROR = "Query processing failed"


class QueueFullError(Exception):
    """Raised by enqueue when the queue is at capacity (mapped to HTTP 503)."""


class QueryJobQueue:
    def __init__(self, ttl_seconds: int | None = None, max_jobs: int | None = None,
                 job_timeout: int | None = None):
        from src.config import settings
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.async_query_ttl_seconds
        self._max_jobs = max_jobs if max_jobs is not None else settings.max_async_query_jobs
        self._job_timeout = job_timeout if job_timeout is not None else settings.async_query_timeout_seconds
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

    def enqueue(self, question: str, username: str, groups: list[str],
                skip_cache: bool = False) -> str:
        self._evict_expired()
        # Cap total tracked jobs so an authenticated caller can't grow the queue
        # without bound (only terminal jobs are TTL-evicted; queued/processing are not).
        if len(self._jobs) >= self._max_jobs:
            raise QueueFullError(f"async query queue is full ({self._max_jobs} jobs)")
        token = str(uuid.uuid4())
        self._jobs[token] = QueryJob(
            token=token, question=question, username=username, groups=groups,
            skip_cache=skip_cache,
        )
        if self._queue is not None:
            self._queue.put_nowait(token)
        return token

    def get_job(self, token: str) -> QueryJob | None:
        self._evict_expired()
        return self._jobs.get(token)

    def update_step(self, token: str, node_name: str, detail: dict | None = None) -> None:
        """Update a job's current step label and append it to the step timeline.
        ``detail`` carries structured node output; a classification detail
        ({"kind": "classification", "data": {...}}) is stored on the job."""
        job = self._jobs.get(token)
        if job:
            job.status = QueryStatus.PROCESSING
            label = STEP_LABELS.get(node_name, node_name)
            job.step = label
            job.steps.append({"step": label, "at": round(time.time() - job.created_at, 2)})
            if detail and detail.get("kind") == "classification":
                job.classification = detail.get("data")

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

    max_parallel: int = 3  # class default; overridden from settings in start_worker

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
        # Respect an instance-level max_parallel if a caller (or test) set one;
        # otherwise take the configured value from settings.
        if self.max_parallel == QueryJobQueue.max_parallel:
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
                # Cap each job so a wedged query (e.g. a hung vLLM call) can't hold
                # a worker slot forever. The ceiling is generous — this mode exists
                # for long queries — and only catches genuinely stuck work.
                result = await asyncio.wait_for(
                    agent_query_streamed(
                        question=job.question, user_groups=job.groups,
                        vector_store=vector_store, schema_registry=schema_registry,
                        metadata_store=metadata_store,
                        step_callback=lambda node, detail=None, _t=token: self.update_step(_t, node, detail),
                        skip_cache=job.skip_cache,
                    ),
                    timeout=self._job_timeout,
                )
                citation_dicts = [
                    {"doc_id": c.doc_id, "filename": c.filename, "doc_type": c.doc_type,
                     "chunk_index": c.chunk_index, "page": c.page, "snippet": c.snippet,
                     "relevance": c.relevance}
                    for c in result.citations
                ]
                self.complete(token, answer=result.answer, citations=citation_dicts,
                              cached=result.cached, cached_query=result.cached_query)
            except asyncio.TimeoutError:
                logger.error(f"Async query {token} timed out after {self._job_timeout}s")
                self.fail(token, "Query timed out")
            except Exception as e:
                # Log the full detail server-side; return only a generic message.
                logger.error(f"Async query {token} failed: {e}\n{traceback.format_exc()}")
                self.fail(token, _GENERIC_ERROR)
            self._queue.task_done()


# Singleton
query_queue = QueryJobQueue()
