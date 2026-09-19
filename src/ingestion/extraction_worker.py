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
MIB = 1024 * 1024


class SpawnedProcess:
    """Small asyncio-compatible wrapper around a fork-free POSIX spawn."""

    def __init__(self, pid: int):
        self.pid = pid
        self.returncode: int | None = None
        self._wait_task: asyncio.Task | None = None

    async def wait(self) -> int:
        if self.returncode is not None:
            return self.returncode
        if self._wait_task is None:
            self._wait_task = asyncio.create_task(asyncio.to_thread(os.waitpid, self.pid, 0))
        _, status = await asyncio.shield(self._wait_task)
        self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode


def spawn_process(command: list[str], log_path: Path, env: dict[str, str]) -> SpawnedProcess:
    """Start the extractor without forking the initialized API process.

    A normal subprocess fork can make a memory-heavy API briefly exceed its
    container budget before exec replaces the child. posix_spawn lets libc use
    a vfork/exec implementation and avoids copying LanceDB/native runtime state.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(log_path, flags, 0o600)
    try:
        actions = [
            (os.POSIX_SPAWN_DUP2, fd, 1),
            (os.POSIX_SPAWN_DUP2, fd, 2),
        ]
        if fd > 2:
            actions.append((os.POSIX_SPAWN_CLOSE, fd))
        spawn = os.posix_spawn if os.path.dirname(command[0]) else os.posix_spawnp
        pid = spawn(command[0], command, env, file_actions=actions, setsid=True)
    finally:
        os.close(fd)
    return SpawnedProcess(pid)


def worker_diagnostic(path: Path, task_dir: Path, max_bytes: int = 4096) -> str:
    """Return a bounded stderr tail without exposing the private task path."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            text = stream.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-12:]).replace(str(task_dir), "<extraction-task>")


def available_memory_bytes() -> int | None:
    """Return the smallest visible system/container memory ceiling."""
    budgets = []
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            budget = int(Path(name).read_text().strip())
            if 0 < budget < 2**60:
                budgets.append(budget)
        except (OSError, ValueError):
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                budgets.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    return min(budgets) if budgets else None


def current_memory_usage() -> int | None:
    """Return total memory charged to this container's cgroup when available."""
    for name in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return int(Path(name).read_text().strip())
        except (OSError, ValueError):
            pass
    return None


def extraction_memory_ceiling(memory_mb: int, baseline: int | None) -> int | None:
    """Calculate a container-usage ceiling that leaves the API headroom."""
    requested = max(128, int(memory_mb)) * MIB
    ceilings = []
    if baseline is not None:
        ceilings.append(baseline + requested)
    total = available_memory_bytes()
    if total is not None:
        # Keep one quarter of the budget, and never less than 512 MiB, outside
        # extraction. On the 4 GiB production LXC this reserves 1 GiB.
        reserve = max(512 * MIB, total // 4)
        ceilings.append(max(128 * MIB, total - reserve))
    return min(ceilings) if ceilings else None


def fail(task_dir: Path, message: str) -> None:
    write_json(task_dir / "status.json", {"state": "failed", "error": message})


async def stop_process(process) -> None:
    # Kill OCR/rendering grandchildren too; they inherit the child's session.
    if process.returncode is None:
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
        memory_baseline = current_memory_usage()
        memory_ceiling = extraction_memory_ceiling(request["memory_mb"], memory_baseline)
        write_json(task_dir / "status.json", {"state": "running", "progress": "Parsing document in isolated worker"})
        command = command or [sys.executable, "-m", "src.ingestion.extraction_worker", "--child", str(task_dir)]
        log_path = task_dir / "worker.log"
        process = spawn_process(
            command,
            log_path,
            {**os.environ, "PYTHONFAULTHANDLER": "1",
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
            usage = current_memory_usage()
            if memory_ceiling is not None and usage is not None and usage > memory_ceiling:
                await stop_process(process)
                fail(task_dir, "Extraction exceeded its memory limit; the API remains available")
                return
            try:
                await asyncio.wait_for(process.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                pass
        if process.returncode:
            reason = f"exit code {process.returncode}"
            if process.returncode < 0:
                reason = signal.Signals(-process.returncode).name
            logger.error("Extractor %s stopped: %s", task_dir.name, reason)
            detail = worker_diagnostic(log_path, task_dir)
            message = f"Document extraction worker stopped ({reason}); the API remains available"
            if detail:
                message += f"\nWorker diagnostic:\n{detail}"
            fail(task_dir, message)
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

        from src.figures.storage import extraction_assets
        with extraction_assets(task_dir / "figures") as notices:
            result = asyncio.run(prepare_document(source, request["filename"], progress))
            result.warnings.extend(notices)
            figures = result.pdf.figure_records if result.pdf else (result.office.figures if result.office else [])
            incomplete = sum(f.analysis_status not in ("complete", "source_extracted") for f in figures)
            if incomplete:
                result.warnings.append(f"{incomplete} stored figure(s) have no completed visual analysis; source context is searchable.")
        write_json(task_dir / "result.json", encode_prepared(result))
    except Exception as exc:
        logger.exception("Document extraction failed")
        fail(task_dir, f"Document extraction failed: {type(exc).__name__}: {str(exc)[:1000]}")


def main():
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3 or sys.argv[1] != "--child":
        raise SystemExit("This module is launched by the ingestion supervisor")
    child(Path(sys.argv[2]))


if __name__ == "__main__":
    main()
