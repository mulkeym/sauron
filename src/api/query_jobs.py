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
