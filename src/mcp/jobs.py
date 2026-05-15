from __future__ import annotations
# src/mcp/jobs.py
from __future__ import annotations
import time
import uuid
from enum import StrEnum

class JobStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"

class JobStore:
    def __init__(self, ttl_seconds: int = 3600):
        self._jobs: dict[str, dict] = {}
        self._ttl = ttl_seconds

    def create(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {"status": JobStatus.PROCESSING, "result": None, "error": None, "created_at": time.time()}
        return job_id

    def get(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def complete(self, job_id: str, result: dict) -> None:
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = JobStatus.COMPLETE
            self._jobs[job_id]["result"] = result

    def fail(self, job_id: str, error: str) -> None:
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = JobStatus.FAILED
            self._jobs[job_id]["error"] = error

    def cleanup_expired(self) -> int:
        now = time.time()
        expired = [jid for jid, j in self._jobs.items() if now - j["created_at"] > self._ttl]
        for jid in expired:
            del self._jobs[jid]
        return len(expired)
