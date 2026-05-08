# tests/test_mcp/test_jobs.py
import pytest
import time
from src.mcp.jobs import JobStore, JobStatus

def test_create_job():
    store = JobStore()
    job_id = store.create()
    assert job_id is not None
    job = store.get(job_id)
    assert job["status"] == JobStatus.PROCESSING

def test_complete_job():
    store = JobStore()
    job_id = store.create()
    store.complete(job_id, result={"answer": "test", "citations": []})
    job = store.get(job_id)
    assert job["status"] == JobStatus.COMPLETE
    assert job["result"]["answer"] == "test"

def test_fail_job():
    store = JobStore()
    job_id = store.create()
    store.fail(job_id, error="Something went wrong")
    job = store.get(job_id)
    assert job["status"] == JobStatus.FAILED
    assert "Something went wrong" in job["error"]

def test_get_nonexistent_job():
    store = JobStore()
    assert store.get("nonexistent") is None

def test_expired_jobs_cleaned():
    store = JobStore(ttl_seconds=0)
    job_id = store.create()
    time.sleep(0.1)
    store.cleanup_expired()
    assert store.get(job_id) is None
