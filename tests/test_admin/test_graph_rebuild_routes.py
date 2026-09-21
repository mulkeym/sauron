from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.admin import routes
from src.ingestion.queue import IngestQueue, IngestStep


def test_admin_rebuild_auth_queue_progress_and_duplicate_guard(monkeypatch):
    q = IngestQueue()
    q.start_worker = AsyncMock()
    monkeypatch.setattr(routes, "get_vector_store", lambda: Mock())
    doc = SimpleNamespace(doc_id='d', filename='guide.pdf', acl_groups=['private'], dataset_id=5, chunk_count=300)
    monkeypatch.setattr(routes, 'ingest_queue', q)
    monkeypatch.setattr(routes, 'get_metadata_store', lambda: SimpleNamespace(list_documents=AsyncMock(return_value=[doc])))
    app = FastAPI(); app.include_router(routes.router)
    with TestClient(app) as client:
        assert client.post('/admin/api/knowledge-graph/rebuild').status_code == 401
        assert client.get('/admin/api/knowledge-graph/rebuild').status_code == 401
        token = routes._create_session()
        client.cookies.set('sauron_session', token)
        try:
            assert client.post('/admin/api/knowledge-graph/rebuild', headers={'origin':'https://untrusted.example'}).status_code == 403
            response = client.post('/admin/api/knowledge-graph/rebuild')
            assert response.status_code == 200 and response.json()['queued'] == 1
            q.start_worker.assert_awaited_once()
            assert client.post('/admin/api/knowledge-graph/rebuild').status_code == 409
            status = client.get('/admin/api/knowledge-graph/rebuild').json()
            assert status['running'] and status['jobs'][0]['filename'] == 'guide.pdf'
            job = q.get_job(response.json()['job_ids'][0])
            q.fail_job(job.job_id, 'Model connection failed')
            status = client.get('/admin/api/knowledge-graph/rebuild').json()
            assert not status['running'] and status['jobs'][0]['error'] == 'Model connection failed'
            assert client.post('/admin/api/knowledge-graph/rebuild').status_code == 200
            status = client.get('/admin/api/knowledge-graph/rebuild').json()
            assert len(status['jobs']) == 1 and not status['jobs'][0]['error']
        finally:
            routes._active_sessions.discard(token)


def test_rebuild_refuses_while_upload_is_active(monkeypatch):
    q = IngestQueue();q.enqueue('new.pdf', '/tmp/new.pdf', [], 'admin')
    monkeypatch.setattr(routes, 'ingest_queue', q)
    app=FastAPI();app.include_router(routes.router)
    token=routes._create_session()
    try:
        with TestClient(app) as client:
            client.cookies.set('sauron_session', token)
            assert client.post('/admin/api/knowledge-graph/rebuild').status_code == 409
    finally:
        routes._active_sessions.discard(token)


def test_admin_delete_reports_graph_failure(monkeypatch):
    store = SimpleNamespace(delete_document=AsyncMock(), delete_entities_for_doc=AsyncMock(),
                            list_documents=AsyncMock(return_value=[SimpleNamespace(doc_id='keep')]))
    monkeypatch.setattr(routes, 'get_metadata_store', lambda: store)
    monkeypatch.setattr(routes, 'get_vector_store', lambda: SimpleNamespace(delete_by_doc_id=Mock()))
    monkeypatch.setattr(routes, 'get_schema_registry', lambda: Mock())
    monkeypatch.setattr('src.ingestion.tabular_ingest.cleanup_spreadsheet_tables', AsyncMock())
    cleanup=AsyncMock(side_effect=RuntimeError('delete failed'))
    monkeypatch.setattr('src.knowledge.graph_rag.reconcile_lightrag_with_metadata', cleanup)
    app=FastAPI();app.include_router(routes.router)
    token=routes._create_session()
    try:
        with TestClient(app) as client:
            client.cookies.set('sauron_session', token)
            response=client.delete('/admin/api/documents/old')
            assert response.status_code == 503
            assert 'knowledge graph cleanup failed' in response.json()['detail']
            cleanup.assert_awaited_once_with({'keep'})
    finally:
        routes._active_sessions.discard(token)


def test_rebuild_dispatches_without_a_prior_upload(monkeypatch):
    import time
    q = IngestQueue()
    doc = SimpleNamespace(doc_id='d', filename='guide.pdf', acl_groups=[], dataset_id=0, chunk_count=1)
    async def process(job, vectors, store):
        q.complete_job(job.job_id, job.doc_id, 1)
    q._process_job = AsyncMock(side_effect=process)
    monkeypatch.setattr(routes, 'ingest_queue', q)
    monkeypatch.setattr(routes, 'get_vector_store', lambda: Mock())
    monkeypatch.setattr(routes, 'get_metadata_store', lambda: SimpleNamespace(list_documents=AsyncMock(return_value=[doc])))
    app=FastAPI();app.include_router(routes.router)
    token=routes._create_session()
    try:
        with TestClient(app) as client:
            client.cookies.set('sauron_session', token)
            assert client.post('/admin/api/knowledge-graph/rebuild').status_code == 200
            for _ in range(50):
                result=client.get('/admin/api/knowledge-graph/rebuild').json()
                if not result['running']: break
                time.sleep(.01)
            assert result['jobs'][0]['status'] == 'complete'
            q._process_job.assert_awaited_once()
    finally:
        routes._active_sessions.discard(token)
