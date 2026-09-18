"""File-backed IPC for isolated document extraction.

Only JSON crosses the process boundary. The worker never opens Sauron's
databases. There is deliberately no fallback to parsing inside the API process.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import weakref
from pathlib import Path


class ExtractionWorkerError(RuntimeError):
    """An isolated extractor failed, exited, or exceeded its limits."""


def write_json(path: Path, value) -> None:
    """Publish complete JSON documents across the process boundary."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def read_json(path: Path, max_bytes: int):
    # Read at most the bound even if another process is still writing the file.
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ExtractionWorkerError("Extraction result exceeds the configured size limit")
    return json.loads(raw)


# One native extractor at a time, even when several indexing jobs are active.
# Initialize per event loop (tests and local clients can create multiple loops).
_slots = weakref.WeakKeyDictionary()


async def extract_in_worker(file_path: Path, filename: str = "", progress_cb=None):
    from src.config import settings
    loop = asyncio.get_running_loop()
    slot = _slots.setdefault(loop, asyncio.Semaphore(1))
    async with slot:
        return await _extract(file_path, filename, progress_cb, settings)


async def _extract(file_path, filename, progress_cb, settings):
    from src.ingestion.prepared import decode_prepared
    from src.ingestion.extraction_worker import run_task

    root = Path(settings.extraction_work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    task_dir = Path(tempfile.mkdtemp(prefix="extract-", dir=root))
    source = task_dir / ("source" + file_path.suffix.lower())
    task = None
    try:
        await asyncio.to_thread(shutil.copyfile, file_path, source)
        request = {
            "source": source.name,
            "filename": filename or file_path.name,
            "settings": settings.model_dump(),
            "timeout": settings.extraction_timeout_seconds,
            "memory_mb": settings.extraction_memory_mb,
            "max_result_bytes": settings.extraction_max_result_mb * 1024 * 1024,
        }
        write_json(task_dir / "request.json", request)
        task = asyncio.create_task(run_task(task_dir))
        last_progress = ""
        while not task.done():
            status_path = task_dir / "status.json"
            if status_path.exists() and progress_cb:
                status = read_json(status_path, 64 * 1024)
                progress = status.get("progress", "")
                if progress and progress != last_progress:
                    progress_cb(progress)
                    last_progress = progress
            await asyncio.sleep(0.1)
        await task
        status = read_json(task_dir / "status.json", 64 * 1024)
        if status["state"] != "complete":
            raise ExtractionWorkerError(status.get("error", "Extraction worker failed"))
        result = await asyncio.to_thread(
            read_json, task_dir / "result.json", request["max_result_bytes"]
        )
        return decode_prepared(result)
    finally:
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(shutil.rmtree, task_dir, True)
