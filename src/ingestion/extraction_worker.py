"""Supervise fresh extractor processes, one file at a time.

The API supervises a child process inside the same container. The supervisor
uses only the standard library; native parser libraries load in the child.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

from src.ingestion.isolation import read_json, write_json

logger = logging.getLogger(__name__)


def fail(task_dir: Path, message: str) -> None:
    write_json(task_dir / "status.json", {"state": "failed", "error": message})


async def stop_process(process) -> None:
    # Kill OCR/rendering grandchildren too; they inherit the child's session.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


async def run_task(task_dir: Path, *, command: list[str] | None = None) -> None:
    """Claim a request and report a terminal status after the child has exited."""
    request_path = task_dir / "request.json"
    try:
        request_path.rename(task_dir / "running.json")
    except FileNotFoundError:
        return
    process = None
    try:
        request = read_json(task_dir / "running.json", 1024 * 1024)
        timeout = max(1, float(request["timeout"]))
        write_json(task_dir / "status.json", {"state": "running", "progress": "Parsing document in isolated worker"})
        command = command or [sys.executable, "-m", "src.ingestion.extraction_worker", "--child", str(task_dir)]
        with (task_dir / "worker.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=log, stderr=log, start_new_session=True,
                env={**os.environ, "PYTHONFAULTHANDLER": "1",
                     "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"},
            )
            started = time.monotonic()
            while process.returncode is None:
                if (task_dir / "cancel").exists():
                    await stop_process(process)
                    fail(task_dir, "Extraction cancelled")
                    return
                if time.monotonic() - started > timeout:
                    await stop_process(process)
                    fail(task_dir, f"Extraction exceeded its {timeout:g}-second time limit")
                    return
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    pass
        if process.returncode:
            reason = f"exit code {process.returncode}"
            if process.returncode < 0:
                reason = signal.Signals(-process.returncode).name
            logger.error("Extractor %s stopped: %s", task_dir.name, reason)
            fail(task_dir, f"Document extraction worker stopped ({reason}); the API remains available")
        elif not (task_dir / "result.json").exists():
            # Normal Python exceptions publish their own diagnostic status.
            if not (task_dir / "status.json").exists() or read_json(task_dir / "status.json", 65536)["state"] != "failed":
                fail(task_dir, "Extraction worker exited without producing a result")
        elif (task_dir / "result.json").stat().st_size > request["max_result_bytes"]:
            fail(task_dir, "Extraction result exceeds the configured size limit")
        else:
            write_json(task_dir / "status.json", {"state": "complete"})
    except asyncio.CancelledError:
        if process is not None:
            await stop_process(process)
        fail(task_dir, "Extraction worker was stopped")
        raise
    except Exception:
        logger.exception("Extraction supervisor failed for %s", task_dir.name)
        fail(task_dir, "Extraction supervisor failed; check the worker logs")
    finally:
        if process is not None:
            # A crashed parser may have left live OCR/rendering descendants.
            await stop_process(process)


def child(task_dir: Path) -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    request = read_json(task_dir / "running.json", 1024 * 1024)
    # Bound temporary files and logs as well as the serialized result.
    file_limit = max(request["max_result_bytes"] * 2, 64 * 1024 * 1024)
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
    try:
        apply_memory_limit(request["memory_mb"])
        from src.config import settings
        for key, value in request["settings"].items():
            if key in type(settings).model_fields:
                setattr(settings, key, value)
        from src.ssl_config import apply_ssl_verify_setting
        apply_ssl_verify_setting()
        source = (task_dir / request["source"]).resolve()
        if source.parent != task_dir.resolve():
            raise ValueError("Invalid extraction source path")
        from src.ingestion.prepared import prepare_document, encode_prepared

        def progress(message):
            write_json(task_dir / "status.json", {"state": "running", "progress": message})

        result = asyncio.run(prepare_document(source, request["filename"], progress))
        write_json(task_dir / "result.json", encode_prepared(result))
    except Exception as exc:
        logger.exception("Document extraction failed")
        fail(task_dir, f"Document extraction failed: {type(exc).__name__}: {str(exc)[:1000]}")


def apply_memory_limit(memory_mb: int) -> None:
    """Hard address-space ceiling on Linux, capped at half the cgroup budget.

    Address space is stricter than RSS: a model that cannot fit fails the file
    rather than letting the extractor allocate the container's entire budget.
    macOS does not reliably enforce RLIMIT_AS; the production container does.
    """
    if sys.platform != "linux":
        return
    import resource
    limit = max(128, int(memory_mb)) * 1024 * 1024
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            budget = int(Path(name).read_text().strip())
            if 0 < budget < 2**60:
                limit = min(limit, budget // 2)
        except (OSError, ValueError):
            pass
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def main():
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3 or sys.argv[1] != "--child":
        raise SystemExit("This module is launched by the ingestion supervisor")
    child(Path(sys.argv[2]))


if __name__ == "__main__":
    main()
