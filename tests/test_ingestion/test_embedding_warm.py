import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.config import settings
from src.ingestion import embedding_isolation as isolated

FAKE = '''
import json,sys,time,os
from pathlib import Path
import numpy as np
root=Path(sys.argv[1])
while True:
    pointer=root/'next.json'
    if not pointer.exists():
        time.sleep(.01); continue
    job=root/json.loads(pointer.read_text())['job']; pointer.unlink()
    req=json.loads((job/'request.json').read_text())
    if req['texts'][0]=='crash':
        print('test native crash',flush=True); os._exit(127)
    if req['texts'][0]=='oom' and req['batch_size']>2:
        (job/'status.json').write_text(json.dumps({'state':'failed','memory_exhausted':True,'error':'Memory allocation failed'}))
        continue
    if req['texts'][0]=='hang': time.sleep(30)
    values=np.asarray([[len(t),req['batch_size'],os.getpid()] for t in req['texts']],dtype=np.float32)
    if req['texts'][0]=='invalid': values[0,0]=float('nan')
    np.save(job/'result.npy',values)
    (job/'tmp.json').write_text(json.dumps({'state':'complete'}))
    (job/'tmp.json').replace(job/'status.json')
'''


@pytest.fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'extraction_work_dir', str(tmp_path/'work'))
    monkeypatch.setattr(settings, 'embedding_batch_size', 8)
    monkeypatch.setattr(settings, 'embedding_worker_timeout_seconds', 5)
    monkeypatch.setattr(settings, 'embedding_worker_idle_seconds', 30)
    monkeypatch.setattr(isolated, 'current_memory_usage', lambda: None)
    monkeypatch.setattr(isolated, 'extraction_memory_ceiling', lambda *args: None)
    w = isolated.EmbeddingWorker(command=[sys.executable, '-c', FAKE])
    yield w
    w.close()
    assert not list((tmp_path/'work').glob('embed-*'))


def test_reuse_honors_batch_above_old_cap_and_preserves_order(worker, monkeypatch):
    first=worker.run(['a','bb'])
    pid=worker.process.pid
    assert [r[:2] for r in first] == [[1,8],[2,8]]
    assert not worker.last_result['reused']
    monkeypatch.setattr(settings, 'embedding_batch_size', 16)
    second=worker.run(['ccc','d'])
    assert [r[:2] for r in second] == [[3,16],[1,16]]
    assert worker.process.pid == pid and worker.last_result['reused']
    assert worker.last_result['passages_per_second'] > 0
    assert not list(worker.directory.glob('job-*'))


def test_concurrent_requests_share_one_worker(worker):
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(worker.run,[['a'],['bb'],['ccc']]))
    assert [r[0][0] for r in results] == [1,2,3]
    assert len({r[0][2] for r in results}) == 1


def test_model_and_thread_config_recycle_worker(worker, monkeypatch):
    worker.run(['a']); previous=worker.process.pid
    monkeypatch.setattr(isolated, 'effective_cpu_threads', lambda: (2,1))
    monkeypatch.setattr(settings, 'embedding_model_name', 'changed-model')
    worker.run(['bb'])
    assert worker.process.pid != previous
    assert not worker.last_result['reused']
    config=json.loads((worker.directory/'config.json').read_text())
    assert config['threads']==2 and config['model']=='changed-model'


@pytest.mark.parametrize('text, match', [('crash','exit code 127'),('invalid','invalid result')])
def test_failure_restarts_on_next_request(worker, text, match):
    with pytest.raises(isolated.EmbeddingWorkerError, match=match): worker.run([text])
    assert worker.process is None
    assert worker.run(['ok'])[0][0] == 2


def test_timeout_kills_worker_then_recovers(worker, monkeypatch):
    monkeypatch.setattr(settings, 'embedding_worker_timeout_seconds', 1)
    with pytest.raises(isolated.EmbeddingWorkerError, match='time limit'): worker.run(['hang'])
    monkeypatch.setattr(settings, 'embedding_worker_timeout_seconds', 5)
    assert worker.run(['ok'])[0][0] == 2


def test_memory_limit_does_not_ratchet_with_warm_requests(worker, monkeypatch):
    worker.run(['first'])
    monkeypatch.setattr(worker, '_over_memory', lambda: True)
    with pytest.raises(isolated.EmbeddingWorkerError, match='memory limit'): worker.run(['next'])
    assert worker.process is None
    os.kill(os.getpid(),0)


def test_idle_releases_model(worker, monkeypatch):
    monkeypatch.setattr(settings,'embedding_worker_idle_seconds',1)
    worker.run(['a'])
    deadline=time.monotonic()+4
    while worker.process is not None and time.monotonic()<deadline: time.sleep(.05)
    assert worker.process is None
    worker.run(['b'])
    assert not worker.last_result['reused']


def test_shutdown_interrupts_active_request(worker):
    with ThreadPoolExecutor(max_workers=1) as pool:
        result=pool.submit(worker.run,['hang'])
        deadline=time.monotonic()+3
        while not worker.process and time.monotonic()<deadline: time.sleep(.01)
        worker.close()
        with pytest.raises(isolated.EmbeddingWorkerError, match='shutting down'): result.result(timeout=2)
    assert worker.process is None


def test_cpu_auto_respects_quota_and_affinity(monkeypatch):
    original=Path.read_text
    def read(path,*a,**kw):
        if str(path)=='/sys/fs/cgroup/cpu.max': return '250000 100000'
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(os,'cpu_count',lambda:32)
    monkeypatch.setattr(os,'sched_getaffinity',lambda _:set(range(8)),raising=False)
    monkeypatch.setattr(settings,'embedding_cpu_threads',0)
    monkeypatch.setattr(settings,'embedding_cpu_interop_threads',1)
    assert isolated.effective_cpu_threads()==(3,1)
    monkeypatch.setattr(settings,'embedding_cpu_threads',2)
    assert isolated.effective_cpu_threads()==(2,1)


def test_memory_retries_smaller_and_remembers_limit(worker, monkeypatch):
    result = worker.run(['oom', 'second', 'third'])
    assert [r[:2] for r in result] == [[3, 2], [6, 2], [5, 2]]
    assert worker.last_result['memory_retries'] == 2
    assert worker.last_result['requested_batch_size'] == 8
    assert settings.embedding_batch_size == 8
    pid = worker.process.pid
    assert worker.run(['next'])[0][1] == 2
    assert worker.process.pid == pid and worker.last_result['memory_retries'] == 0
    monkeypatch.setattr(settings, 'embedding_batch_size', 4)
    assert worker.run(['next'])[0][1] == 4


def test_memory_retries_share_one_deadline(worker, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(isolated.time, 'monotonic', lambda: clock[0])
    batches = []
    def attempt(texts, batch, *, deadline):
        assert deadline == 105.
        batches.append(batch)
        clock[0] += 3
        raise isolated.EmbeddingMemoryError('memory limit')
    monkeypatch.setattr(worker, '_run_attempt', attempt)
    with pytest.raises(isolated.EmbeddingWorkerError, match='time limit during memory recovery'):
        worker.run(['oom'])
    assert batches == [8, 4]


def test_non_memory_errors_do_not_retry(worker, monkeypatch):
    calls = []
    def attempt(*args, **kwargs):
        calls.append(1)
        raise isolated.EmbeddingWorkerError('invalid result')
    monkeypatch.setattr(worker, '_run_attempt', attempt)
    with pytest.raises(isolated.EmbeddingWorkerError, match='invalid result'):
        worker.run(['invalid'])
    assert len(calls) == 1
