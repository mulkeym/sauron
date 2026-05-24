"""Tests for IngestQueue.has_active_jobs — used to block KG purge during ingestion."""
from src.ingestion.queue import IngestJob, IngestQueue, IngestStep


def _job(step: IngestStep) -> IngestJob:
    return IngestJob(
        job_id=f"j-{step}", filename="f.xls", file_path="/tmp/f.xls",
        acl_groups=[], uploaded_by="test", step=step,
    )


def test_no_jobs_is_inactive():
    q = IngestQueue()
    assert q.has_active_jobs() is False


def test_only_terminal_jobs_is_inactive():
    q = IngestQueue()
    q._jobs = {j.job_id: j for j in (_job(IngestStep.COMPLETE), _job(IngestStep.FAILED))}
    assert q.has_active_jobs() is False


def test_queued_job_is_active():
    q = IngestQueue()
    q._jobs = {"a": _job(IngestStep.QUEUED)}
    assert q.has_active_jobs() is True


def test_mid_processing_job_is_active():
    q = IngestQueue()
    q._jobs = {
        "done": _job(IngestStep.COMPLETE),
        "busy": _job(IngestStep.EXTRACTING_ENTITIES),
    }
    assert q.has_active_jobs() is True
