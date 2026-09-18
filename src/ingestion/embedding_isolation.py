"""Run the local embedding model outside the API process."""
from __future__ import annotations

import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

from src.ingestion.extraction_worker import (
    current_memory_usage,
    extraction_memory_ceiling,
    spawn_process,
    worker_diagnostic,
)
from src.ingestion.isolation import read_json, write_json

logger = logging.getLogger(__name__)


class EmbeddingWorkerError(RuntimeError):
    """The isolated local embedding process failed or exceeded a limit."""


def _stop(process) -> None:
    if process.returncode is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            _, status = os.waitpid(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            pass


def embed_local_in_worker(
    texts: list[str], batch_size: int,
    *, command: list[str] | None = None,
) -> list[list[float]]:
    """Embed text in a disposable process and return a bounded NumPy result."""
    if not texts:
        return []
    from src.config import settings
    import numpy as np

    root = Path(settings.extraction_work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    task_dir = Path(tempfile.mkdtemp(prefix="embed-", dir=root))
    process = None
    try:
        # Four is deliberately conservative for a 4 GiB deployment. The admin
        # setting can lower this further but cannot bypass the safety ceiling.
        safe_batch = max(1, min(int(batch_size or settings.embedding_batch_size),
                                int(settings.embedding_batch_size), 4))
        max_result_bytes = max(64, settings.extraction_max_result_mb * 4) * 1024 * 1024
        write_json(task_dir / "request.json", {
            "texts": texts,
            "batch_size": safe_batch,
            "max_result_bytes": max_result_bytes,
        })
        baseline = current_memory_usage()
        ceiling = extraction_memory_ceiling(settings.extraction_memory_mb, baseline)
        worker_command = command or [
            sys.executable, "-m", "src.ingestion.embedding_isolation", "--child"
        ]
        process = spawn_process(
            [*worker_command, str(task_dir)],
            task_dir / "worker.log",
            {**os.environ, "PYTHONFAULTHANDLER": "1", "OMP_NUM_THREADS": "1",
             "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
             "TOKENIZERS_PARALLELISM": "false"},
        )
        started = time.monotonic()
        while process.returncode is None:
            pid, status = os.waitpid(process.pid, os.WNOHANG)
            if pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                break
            if time.monotonic() - started > settings.extraction_timeout_seconds:
                _stop(process)
                raise EmbeddingWorkerError(
                    f"Embedding exceeded its {settings.extraction_timeout_seconds:g}-second time limit; "
                    "the API remains available"
                )
            usage = current_memory_usage()
            if ceiling is not None and usage is not None and usage > ceiling:
                _stop(process)
                raise EmbeddingWorkerError(
                    "Embedding exceeded its memory limit; the API remains available"
                )
            time.sleep(0.05)

        status_path = task_dir / "status.json"
        if process.returncode:
            reason = f"exit code {process.returncode}"
            if process.returncode < 0:
                reason = signal.Signals(-process.returncode).name
            detail = worker_diagnostic(task_dir / "worker.log", task_dir)
            message = f"Embedding worker stopped ({reason}); the API remains available"
            if detail:
                message += f"\nWorker diagnostic:\n{detail}"
            raise EmbeddingWorkerError(message)
        if not status_path.exists():
            raise EmbeddingWorkerError("Embedding worker exited without a status")
        status = read_json(status_path, 64 * 1024)
        if status.get("state") != "complete":
            raise EmbeddingWorkerError(status.get("error", "Embedding worker failed"))
        result_path = task_dir / "result.npy"
        if not result_path.exists() or result_path.stat().st_size > max_result_bytes:
            raise EmbeddingWorkerError("Embedding result exceeds the configured size limit")
        vectors = np.load(result_path, allow_pickle=False)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise EmbeddingWorkerError("Embedding worker returned an invalid result shape")
        settings.embedding_dimension = int(vectors.shape[1])
        return vectors.tolist()
    finally:
        if process is not None and process.returncode is None:
            _stop(process)
        import shutil
        shutil.rmtree(task_dir, ignore_errors=True)


def child(task_dir: Path) -> None:
    import resource
    import numpy as np

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        request_path = task_dir / "request.json"
        request = read_json(request_path, max(1024 * 1024, request_path.stat().st_size))
        from src.ingestion.embedder import _embed_via_local_direct

        vectors = _embed_via_local_direct(
            request["texts"], batch_size=int(request["batch_size"])
        )
        result_path = task_dir / "result.npy"
        with result_path.open("wb") as stream:
            np.save(stream, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
        if result_path.stat().st_size > int(request["max_result_bytes"]):
            result_path.unlink(missing_ok=True)
            raise ValueError("Embedding result exceeds the configured size limit")
        write_json(task_dir / "status.json", {"state": "complete"})
    except Exception as exc:
        logger.exception("Embedding worker failed")
        write_json(task_dir / "status.json", {
            "state": "failed",
            "error": f"Embedding failed: {type(exc).__name__}: {str(exc)[:1000]}",
        })


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3 or sys.argv[1] != "--child":
        raise SystemExit("This module is launched by the embedding supervisor")
    child(Path(sys.argv[2]))


if __name__ == "__main__":
    main()
