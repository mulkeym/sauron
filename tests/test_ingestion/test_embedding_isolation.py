import os
import sys
from pathlib import Path

import pytest

from src.config import settings
from src.ingestion import embedding_isolation
from src.ingestion.embedding_isolation import EmbeddingWorkerError


@pytest.fixture(autouse=True)
def isolated_work_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "extraction_work_dir", str(tmp_path / "work"))
    monkeypatch.setattr(settings, "extraction_timeout_seconds", 5)
    monkeypatch.setattr(settings, "extraction_max_result_mb", 32)
    monkeypatch.setattr(settings, "embedding_batch_size", 4)


def test_embedding_worker_round_trip():
    script = (
        "import json,numpy as np,sys; from pathlib import Path; "
        "p=Path(sys.argv[1]); r=json.loads((p/'request.json').read_text()); "
        "np.save(p/'result.npy',np.asarray([[i,i+1] for i in range(len(r['texts']))],dtype=np.float32)); "
        "(p/'status.json').write_text(json.dumps({'state':'complete'}))"
    )
    result = embedding_isolation.embed_local_in_worker(
        ["one", "two"], 64, command=[sys.executable, "-c", script]
    )
    assert result == [[0.0, 1.0], [1.0, 2.0]]
    assert settings.embedding_dimension == 2
    assert not list(Path(settings.extraction_work_dir).glob("embed-*"))


def test_embedding_worker_exit_includes_diagnostic():
    command = [sys.executable, "-c",
               "import sys; print('model loader failed safely',file=sys.stderr); raise SystemExit(127)"]
    with pytest.raises(EmbeddingWorkerError) as error:
        embedding_isolation.embed_local_in_worker(["one"], 1, command=command)
    assert "exit code 127" in str(error.value)
    assert "model loader failed safely" in str(error.value)


def test_embedding_memory_pressure_kills_only_worker(monkeypatch):
    mib = 1024 * 1024
    readings = iter([100 * mib, 900 * mib])
    monkeypatch.setattr(embedding_isolation, "current_memory_usage",
                        lambda: next(readings, 900 * mib))
    monkeypatch.setattr(embedding_isolation, "extraction_memory_ceiling",
                        lambda memory_mb, baseline: 512 * mib)
    with pytest.raises(EmbeddingWorkerError, match="memory limit"):
        embedding_isolation.embed_local_in_worker(
            ["one"], 1,
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
        )
    os.kill(os.getpid(), 0)


def test_local_embedder_delegates_to_isolated_worker(monkeypatch):
    from src.ingestion import embedder

    captured = {}
    monkeypatch.setattr(
        embedding_isolation,
        "embed_local_in_worker",
        lambda texts, batch: captured.update(texts=texts, batch=batch) or [[0.1, 0.2]],
    )
    assert embedder._embed_via_local(["hello"], 64) == [[0.1, 0.2]]
    assert captured == {"texts": ["hello"], "batch": 64}
