"""Exercise real child exits, hung parsers, cancellation and healthy follow-ups."""
import asyncio
import os
import signal
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.config import settings
from src.ingestion import extraction_worker
from src.ingestion.isolation import ExtractionWorkerError, extract_in_worker, read_json, write_json
from src.ingestion.prepared import PreparedDocument, decode_prepared, encode_prepared, prepare_document
from src.ingestion.queue import IngestQueue, IngestStep

FIXTURES = Path(__file__).parents[1] / "fixtures" / "pdf"


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "extraction_work_dir", str(tmp_path / "work"))
    monkeypatch.setattr(settings, "figure_extraction_enabled", False)
    monkeypatch.setattr(settings, "extraction_timeout_seconds", 20)


def request_dir(tmp_path, timeout=5, max_result_bytes=1000):
    tmp_path.mkdir(parents=True, exist_ok=True)
    write_json(tmp_path / "request.json", {
        "timeout": timeout,
        "memory_mb": 4096,
        "max_result_bytes": max_result_bytes,
    })
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("sig", [signal.SIGSEGV, signal.SIGKILL, signal.SIGABRT])
async def test_native_death_is_contained_and_next_file_parses(tmp_path, sig):
    task_dir = request_dir(tmp_path / "bad")
    await extraction_worker.run_task(task_dir, command=[sys.executable, "-c",
        f"import os,signal,resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); os.kill(os.getpid(),{int(sig)})"])
    status = read_json(task_dir / "status.json", 65536)
    assert status["state"] == "failed"
    assert sig.name in status["error"]
    source = tmp_path / "healthy.txt"
    source.write_text("This file still imports after a native parser crash.")
    result = await extract_in_worker(source, "healthy.txt")
    assert "still imports" in result.parsed.text
    assert not list(Path(settings.extraction_work_dir).glob("extract-*"))


@pytest.mark.asyncio
async def test_hung_child_times_out_without_blocking_event_loop(tmp_path):
    task_dir = request_dir(tmp_path / "hung", timeout=1)
    task = asyncio.create_task(extraction_worker.run_task(task_dir, command=[
        sys.executable, "-c", "import time; time.sleep(30)"]))
    ticks = 0
    while not task.done():
        ticks += 1
        await asyncio.sleep(0.05)
    await task
    assert ticks >= 10
    assert "time limit" in read_json(task_dir / "status.json", 65536)["error"]


@pytest.mark.asyncio
async def test_cancelling_upload_reaps_child(tmp_path, monkeypatch):
    original = extraction_worker.spawn_process
    children = []

    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(extraction_worker, "spawn_process", spawn)
    task_dir = request_dir(tmp_path / "cancel")
    task = asyncio.create_task(extraction_worker.run_task(task_dir, command=[
        sys.executable, "-c", "import time; time.sleep(30)"]))
    while not children:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert children[0].returncode == -signal.SIGKILL
    with pytest.raises(ProcessLookupError):
        os.kill(children[0].pid, 0)


@pytest.mark.asyncio
async def test_worker_uses_fork_free_spawn(tmp_path, monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("asyncio subprocess would fork the initialized API")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    task_dir = request_dir(tmp_path / "spawn")
    await extraction_worker.run_task(task_dir, command=[sys.executable, "-c", "pass"])
    assert read_json(task_dir / "status.json", 65536)["state"] == "failed"


@pytest.mark.asyncio
async def test_native_exit_includes_bounded_worker_diagnostic(tmp_path):
    task_dir = request_dir(tmp_path / "diagnostic")
    await extraction_worker.run_task(task_dir, command=[
        sys.executable, "-c", "import sys; print('loader failed safely', file=sys.stderr); raise SystemExit(127)"])
    error = read_json(task_dir / "status.json", 65536)["error"]
    assert "exit code 127" in error
    assert "loader failed safely" in error
    assert str(task_dir) not in error


@pytest.mark.asyncio
async def test_container_memory_pressure_stops_worker(tmp_path, monkeypatch):
    mib = 1024 * 1024
    readings = iter([100 * mib, 900 * mib])
    monkeypatch.setattr(extraction_worker, "current_memory_usage", lambda: next(readings, 900 * mib))
    monkeypatch.setattr(extraction_worker, "available_memory_bytes", lambda: 1024 * mib)
    task_dir = request_dir(tmp_path / "memory")
    await extraction_worker.run_task(task_dir, command=[
        sys.executable, "-c", "import time; time.sleep(30)"])
    error = read_json(task_dir / "status.json", 65536)["error"]
    assert "memory limit" in error
    assert "API remains available" in error


@pytest.mark.asyncio
async def test_oversized_result_is_rejected(tmp_path):
    task_dir = request_dir(tmp_path / "large", max_result_bytes=10)
    output = str(task_dir / "result.json")
    await extraction_worker.run_task(task_dir, command=[
        sys.executable, "-c", f"from pathlib import Path; Path({output!r}).write_text('x'*100)"])
    assert "size limit" in read_json(task_dir / "status.json", 65536)["error"]


@pytest.mark.asyncio
async def test_pdf_tables_survive_process_and_json_boundary():
    result = await extract_in_worker(FIXTURES / "two_page_table.pdf", "pay.pdf")
    assert result.parsed.filename == "pay.pdf"
    assert result.pdf is not None
    assert result.pdf.table_grids
    assert any("O-1" in str(grid.rows) for grid in result.pdf.table_grids)


def test_figure_provenance_roundtrips():
    from src.ingestion.parser import ParsedDocument, DocumentBlock, FigurePlacement
    from src.ingestion.pdf_extract import ExtractedPdf, ProseBlock
    from src.ingestion.figure_extract import FigureRecord
    from src.ingestion.tabular import SheetGrid
    prepared = PreparedDocument(ParsedDocument("figure.pdf", "pdf", "hello", blocks=[
        DocumentBlock("figure", 1, figure=FigurePlacement("f1", "r1", 1, bbox=(1, 2, 3, 4)))
    ]), pdf=ExtractedPdf(
        prose_blocks=[ProseBlock("prose", 0)],
        table_grids=[SheetGrid("Pay", [["Grade", "Pay"], ["GS-12", 100]])],
        figure_records=[FigureRecord("f1", "A diagram", "network", page=0, bbox=(1, 2, 3, 4))],
    ))
    import json
    assert decode_prepared(json.loads(json.dumps(encode_prepared(prepared)))) == prepared


@pytest.mark.asyncio
async def test_table_exception_preserves_flat_text_inside_worker(monkeypatch):
    def bad_table(path):
        raise ValueError("unsupported table layout")
    monkeypatch.setattr("src.ingestion.pdf_extract.extract_pdf", bad_table)
    result = await prepare_document(FIXTURES / "two_page_table.pdf", "pay.pdf")
    assert result.pdf is None
    assert result.parsed.text
    assert "unsupported table layout" in result.warnings[0]


@pytest.mark.asyncio
async def test_queue_reports_parser_death_and_health_remains_available(tmp_path, monkeypatch):
    from src.main import app
    real_run = extraction_worker.run_task

    async def run(task_dir):
        request = read_json(task_dir / "request.json", 1024 * 1024)
        if request["filename"] == "bad.pdf":
            return await real_run(task_dir, command=[sys.executable, "-c",
                "import os,signal,time; time.sleep(0.4); os.kill(os.getpid(),signal.SIGKILL)"])
        return await real_run(task_dir)

    monkeypatch.setattr(extraction_worker, "run_task", run)
    queue = IngestQueue()
    queue._cleanup_failed_job = AsyncMock()

    async def process(job, *stores):
        await extract_in_worker(Path(job.file_path), job.filename)
        queue.complete_job(job.job_id, "doc-ok", 1)

    queue._process_job = process
    monkeypatch.setattr(settings, "max_parallel_ingestion", 1)
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"bad PDF")
    good = tmp_path / "good.txt"
    good.write_text("Healthy follow-up file")
    bad_id = queue.enqueue("bad.pdf", str(bad), [], "tester")
    good_id = queue.enqueue("good.txt", str(good), [], "tester")
    await queue.start_worker(None, None)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for _ in range(5):
                response = await asyncio.wait_for(client.get("/api/health", headers={"X-API-Key": "test-key-1"}), 0.5)
                assert response.json() == {"status": "ok"}
                await asyncio.sleep(0.1)
        await asyncio.wait_for(queue._queue.join(), 20)
        assert queue.get_job(bad_id).step == IngestStep.FAILED
        assert "SIGKILL" in queue.get_job(bad_id).error
        assert queue.get_job(good_id).step == IngestStep.COMPLETE
        queue._cleanup_failed_job.assert_awaited_once()
    finally:
        await queue.stop_worker()


def test_nested_lxc_memory_ceiling_reserves_api_headroom(monkeypatch):
    def content(path):
        if str(path) == "/proc/meminfo":
            return "MemTotal:        4194304 kB\nMemFree:         123 kB\n"
        return "max"

    monkeypatch.setattr(Path, "read_text", content)
    assert extraction_worker.available_memory_bytes() == 4 * 1024**3
    assert extraction_worker.extraction_memory_ceiling(4096, 600 * 1024**2) == 3 * 1024**3


def test_operator_memory_limit_can_be_lower_than_system_ceiling(monkeypatch):
    monkeypatch.setattr(extraction_worker, "available_memory_bytes", lambda: 8 * 1024**3)
    assert extraction_worker.extraction_memory_ceiling(512, 600 * 1024**2) == 1112 * 1024**2


@pytest.mark.asyncio
async def test_early_parser_failure_does_not_initialize_vector_storage(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ingestion.isolation.extract_in_worker",
                        AsyncMock(side_effect=ExtractionWorkerError("SIGSEGV")))
    monkeypatch.setattr(settings, "max_parallel_ingestion", 1)
    source = tmp_path / "bad.pdf"
    source.write_bytes(b"bad")
    metadata = AsyncMock()
    metadata.find_by_content_hash.return_value = None
    metadata.list_documents.return_value = []
    vectors = MagicMock()
    queue = IngestQueue()
    job_id = queue.enqueue("bad.pdf", str(source), [], "tester")
    await queue.start_worker(vectors, metadata)
    try:
        await asyncio.wait_for(queue._queue.join(), 5)
        assert queue.get_job(job_id).step == IngestStep.FAILED
        assert not vectors.mock_calls
        metadata.add_document.assert_not_awaited()
        metadata.delete_document.assert_not_awaited()
    finally:
        await queue.stop_worker()


@pytest.mark.asyncio
async def test_large_upload_is_spooled_in_bounded_reads():
    from src.ingestion.uploads import save_upload

    class Upload:
        filename = "large.pdf"
        remaining = 3 * 1024 * 1024 + 17
        async def read(self, size):
            assert 0 < size <= 1024 * 1024
            chunk = b"x" * min(size, self.remaining)
            self.remaining -= len(chunk)
            return chunk

    path = await save_upload(Upload())
    try:
        assert path.stat().st_size == 3 * 1024 * 1024 + 17
    finally:
        path.unlink()
