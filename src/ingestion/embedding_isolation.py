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


class EmbeddingMemoryError(EmbeddingWorkerError):
    """A confirmed memory limit or allocator exhaustion; safe to retry smaller."""


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


def _embed_once(
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
        safe_batch = max(1, min(int(batch_size or settings.embedding_batch_size),
                                int(settings.embedding_batch_size)))
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
            "memory_exhausted": isinstance(exc, MemoryError) or (
                "DefaultCPUAllocator" in str(exc) and "allocate memory" in str(exc)),
            "error": f"Embedding failed: {type(exc).__name__}: {str(exc)[:1000]}",
        })


def effective_cpu_threads():
    """Respect both CPU affinity and Linux container quotas (including v1)."""
    import math
    from src.config import settings
    available = os.cpu_count() or 1
    if hasattr(os, 'sched_getaffinity'):
        available = min(available, len(os.sched_getaffinity(0)) or 1)
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            available = min(available, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError, ZeroDivisionError):
        try:
            quota = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
            period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
            if quota > 0:
                available = min(available, max(1, math.ceil(quota / period)))
        except (OSError, ValueError, ZeroDivisionError):
            pass
    return (min(settings.embedding_cpu_threads or available, available),
            min(settings.embedding_cpu_interop_threads, available))


class EmbeddingWorker:
    """One serialized model per API process, supervised even while idle.

    Only JSON and bounded NumPy arrays cross the process boundary. Each request
    has a private directory; no database handles or credentials are sent in IPC.
    """
    def __init__(self, command=None):
        import threading
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.process = None
        self.directory = None
        self.signature = None
        self.ceiling = None
        self.memory_mb = 0
        self.idle_seconds = 0
        self.last_used = time.monotonic()
        self.active = False
        self.last_result = {}
        self.adaptive_signature = None
        self.adaptive_batch_size = None
        self.memory_limit_detail = ""
        self.command = command
        self.reaper = threading.Thread(target=self._reap, name='embedding-supervisor', daemon=True)
        self.reaper.start()

    def _poll(self):
        if self.process and self.process.returncode is None:
            try:
                pid, status = os.waitpid(self.process.pid, os.WNOHANG)
                if pid:
                    self.process.returncode = os.waitstatus_to_exitcode(status)
            except ChildProcessError:
                self.process.returncode = -1
        return self.process.returncode if self.process else None

    def _over_memory(self):
        usage = current_memory_usage()
        if self.ceiling is not None and usage is not None and usage > self.ceiling:
            self.memory_limit_detail = f"Container usage {usage / 1024**2:.0f} MiB exceeded the {self.ceiling / 1024**2:.0f} MiB safety ceiling."
            return True
        # Also bound the resident model process itself: container baseline must
        # not allow a warm model to grow its allowance on each request.
        if self.process:
            try:
                for line in Path(f'/proc/{self.process.pid}/status').read_text().splitlines():
                    if line.startswith('VmRSS:'):
                        rss_mb = int(line.split()[1]) / 1024
                        if rss_mb > self.memory_mb:
                            self.memory_limit_detail = f"Worker resident memory {rss_mb:.0f} MiB exceeded its {self.memory_mb} MiB budget."
                            return True
                        return False
            except (OSError, ValueError):
                pass
        return False

    def _dispose(self):
        import shutil
        if self.process:
            _stop(self.process)
        if self.directory:
            shutil.rmtree(self.directory, ignore_errors=True)
        self.process = self.directory = self.signature = None

    def _reap(self):
        while not self.closed.wait(0.5):
            if not self.lock.acquire(blocking=False):
                continue
            try:
                if self.process and (self._poll() is not None or self._over_memory()
                        or time.monotonic() - self.last_used >= self.idle_seconds):
                    self._dispose()
            finally:
                self.lock.release()

    def close(self):
        self.closed.set()
        with self.lock:
            self._dispose()

    def run(self, texts, batch_size=0):
        from src.config import settings
        queued = time.monotonic()
        with self.lock:
            started = time.monotonic()
            timeout = settings.embedding_worker_timeout_seconds
            deadline = started + timeout
            signature = (settings.embedding_model_name, settings.embedding_worker_memory_mb,
                         settings.embedding_batch_size, effective_cpu_threads(), settings.extraction_work_dir)
            if self.adaptive_signature != signature:
                self.adaptive_signature = signature
                self.adaptive_batch_size = None
            requested = max(1, min(int(batch_size or settings.embedding_batch_size), settings.embedding_batch_size))
            batch = min(requested, self.adaptive_batch_size or requested)
            retries = 0
            while True:
                try:
                    result = self._run_attempt(texts, batch, deadline=deadline)
                    elapsed = time.monotonic() - started
                    self.last_result.update(requested_batch_size=requested, memory_retries=retries,
                        seconds=round(elapsed, 3), queue_seconds=round(started - queued, 3),
                        passages_per_second=round(len(texts) / max(elapsed, .001), 2))
                    logger.info('Local embedding request complete: %s', self.last_result)
                    return result
                except EmbeddingMemoryError as exc:
                    if batch <= 1:
                        raise EmbeddingMemoryError(
                            f'Embedding cannot fit within its memory limits even at batch size 1. {exc} '
                            'Check container headroom and the embedding worker memory budget. The API remains available'
                        ) from exc
                    if time.monotonic() >= deadline:
                        raise EmbeddingWorkerError(f'Embedding exceeded its {timeout}-second time limit during memory recovery; the API remains available') from exc
                    batch = max(1, batch // 2)
                    self.adaptive_batch_size = batch
                    retries += 1
                    logger.warning('Embedding memory recovery: retry %s at batch %s (requested %s). %s',
                                   retries, batch, requested, exc)

    def _run_attempt(self, texts, batch_size=0, *, deadline=None):
        import numpy as np
        from src.config import settings
        queued = time.monotonic()
        with self.lock:
            if self.closed.is_set():
                raise EmbeddingWorkerError('Embedding worker is shutting down')
            started = time.monotonic()
            threads, interop = effective_cpu_threads()
            config = dict(model=settings.embedding_model_name, threads=threads, interop=interop,
                          memory_mb=settings.embedding_worker_memory_mb,
                          idle_seconds=settings.embedding_worker_idle_seconds)
            signature = (tuple(config.items()), str(Path(settings.extraction_work_dir).resolve()))
            batch = max(1, min(int(batch_size or settings.embedding_batch_size), settings.embedding_batch_size))
            timeout = settings.embedding_worker_timeout_seconds
            deadline = deadline if deadline is not None else started + timeout
            if started >= deadline:
                raise EmbeddingWorkerError(f'Embedding exceeded its {timeout}-second time limit; the API remains available')
            maximum = max(64, settings.extraction_max_result_mb * 4) * 1024 * 1024
            self.active = True
            job_dir = None
            reused = False
            try:
                if self.process and (self.signature != signature or self._poll() is not None):
                    self._dispose()
                reused = self.process is not None
                if not reused:
                    root = Path(settings.extraction_work_dir).resolve()
                    root.mkdir(parents=True, exist_ok=True)
                    self.directory = Path(tempfile.mkdtemp(prefix='embed-warm-', dir=root))
                    self.signature = signature
                    self.memory_mb = config['memory_mb']
                    self.idle_seconds = config['idle_seconds']
                    self.ceiling = extraction_memory_ceiling(self.memory_mb, current_memory_usage())
                    write_json(self.directory / 'config.json', {**config, 'parent_pid': os.getpid()})
                    cmd = self.command or [sys.executable, '-m', 'src.ingestion.embedding_isolation', '--serve']
                    self.process = spawn_process([*cmd, str(self.directory)], self.directory / 'worker.log',
                        {**os.environ, 'PYTHONFAULTHANDLER': '1', 'OMP_NUM_THREADS': str(threads),
                         'MKL_NUM_THREADS': str(threads), 'OPENBLAS_NUM_THREADS': str(threads),
                         'TOKENIZERS_PARALLELISM': 'false'})
                job_dir = Path(tempfile.mkdtemp(prefix='job-', dir=self.directory))
                write_json(job_dir / 'request.json', {'texts': texts, 'batch_size': batch, 'max_result_bytes': maximum})
                write_json(self.directory / 'next.json', {'job': job_dir.name})
                status_path = job_dir / 'status.json'
                while True:
                    if self.closed.is_set():
                        raise EmbeddingWorkerError('Embedding worker is shutting down; the API remains available')
                    if time.monotonic() > deadline:
                        raise EmbeddingWorkerError(f'Embedding exceeded its {timeout}-second time limit; the API remains available')
                    if self._over_memory():
                        raise EmbeddingMemoryError('Embedding exceeded its memory limit. ' + self.memory_limit_detail)
                    code = self._poll()
                    if code is not None:
                        detail = worker_diagnostic(self.directory / 'worker.log', self.directory)
                        raise EmbeddingWorkerError(f'Embedding worker stopped (exit code {code}); the API remains available\n{detail}')
                    if status_path.exists():
                        break
                    time.sleep(0.05)
                status = read_json(status_path, 64 * 1024)
                if status.get('state') != 'complete':
                    error_type = EmbeddingMemoryError if status.get('memory_exhausted') else EmbeddingWorkerError
                    raise error_type(status.get('error', 'Embedding worker failed'))
                path = job_dir / 'result.npy'
                if not path.exists() or path.stat().st_size > maximum:
                    raise EmbeddingWorkerError('Embedding result exceeds the configured size limit')
                # mmap lets us check dimensions/size before allocating the Python result.
                vectors = np.load(path, allow_pickle=False, mmap_mode='r')
                if (vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] < 1
                        or vectors.dtype != np.float32 or vectors.nbytes > maximum or not np.isfinite(vectors).all()):
                    raise EmbeddingWorkerError('Embedding worker returned an invalid result')
                result = vectors.tolist()
                settings.embedding_dimension = int(vectors.shape[1])
                del vectors
                elapsed = time.monotonic() - started
                self.last_result = dict(passages=len(texts), batch_size=batch, threads=threads, interop=interop,
                    seconds=round(elapsed, 3), passages_per_second=round(len(texts) / max(elapsed, .001), 2),
                    queue_seconds=round(started-queued, 3), reused=reused)
                return result
            except Exception:
                self._dispose()
                raise
            finally:
                import shutil
                if job_dir:
                    shutil.rmtree(job_dir, ignore_errors=True)
                self.last_used = time.monotonic()
                self.active = False


_worker = None
import threading
_worker_guard = threading.Lock()


def embed_local_in_worker(texts, batch_size=0, *, command=None):
    if not texts:
        return []
    # Preserve the disposable test/diagnostic entry point; normal embedding uses
    # the supervised, serialized worker below.
    if command is not None:
        return _embed_once(texts, batch_size, command=command)
    global _worker
    with _worker_guard:
        if _worker is None or _worker.closed.is_set():
            _worker = EmbeddingWorker()
        worker = _worker
    return worker.run(texts, batch_size)


def shutdown_embedding_worker():
    global _worker
    with _worker_guard:
        if _worker is not None:
            _worker.close()
            _worker = None


def embedding_worker_status():
    threads, interop = effective_cpu_threads()
    worker = _worker
    return {'state': 'busy' if worker and worker.active else 'warm' if worker and worker.process else 'stopped',
            'effective_threads': threads, 'effective_interop_threads': interop,
            'last_request': worker.last_result if worker else {}}


def serve(directory):
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    config = read_json(directory / 'config.json', 65536)
    if sys.platform == 'linux':
        import ctypes
        # Terminate even during native inference if the API process disappears.
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot configure embedding parent-death signal')
        if os.getppid() != config['parent_pid']:
            return
    from src.config import settings
    settings.embedding_model_name = config['model']
    settings.embedding_cpu_threads = config['threads']
    settings.embedding_cpu_interop_threads = config['interop']
    # Parent supervises idle expiry and memory. Parent death releases the model
    # even after an ungraceful application exit (once the current encode ends).
    while os.getppid() == config['parent_pid']:
        pointer = directory / 'next.json'
        if pointer.exists():
            task = read_json(pointer, 65536)
            pointer.unlink()
            name = task['job']
            if not isinstance(name, str) or not name.startswith('job-') or Path(name).name != name:
                raise ValueError('Invalid embedding request directory')
            child(directory / name)
        time.sleep(.05)


import atexit
atexit.register(shutdown_embedding_worker)

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3 or sys.argv[1] not in {"--child", "--serve"}:
        raise SystemExit("This module is launched by the embedding supervisor")
    (serve if sys.argv[1] == "--serve" else child)(Path(sys.argv[2]))


if __name__ == "__main__":
    main()
