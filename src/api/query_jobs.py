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


# Singleton
query_queue = QueryJobQueue()
