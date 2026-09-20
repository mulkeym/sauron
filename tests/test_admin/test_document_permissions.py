import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.admin import routes
from src.config import settings
from src.db.metadata import MetadataStore
from src.retrieval.vector_store import VectorStore
from src.retrieval.models import ChunkMetadata


def test_admin_permission_edit_updates_real_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'lancedb_path', str(tmp_path / 'vectors'))
    monkeypatch.setattr(settings, 'embedding_dimension', 3)
    metadata = MetadataStore(f'sqlite+aiosqlite:///{tmp_path}/metadata.db')
    async def setup():
        await metadata.init()
        await metadata.add_document('d', 'sdwan.pdf', 'pdf', ['old'], 1, 'tester')
    asyncio.run(setup())
    vectors = VectorStore()
    vectors.upsert(['Government SD-WAN supports FIPS'], [[1., 0., 0.]],
                   [ChunkMetadata(doc_id='d', filename='sdwan.pdf', doc_type='pdf',
                                  chunk_index=0, start_char=0, acl_groups=['old'])])
    monkeypatch.setattr(routes, 'get_metadata_store', lambda: metadata)
    monkeypatch.setattr(routes, 'get_vector_store', lambda: vectors)
    app = FastAPI(); app.include_router(routes.router)
    with TestClient(app) as client:
        assert client.put('/admin/api/documents/d', data={'acl_groups': 'engineering'}).status_code == 401
        client.cookies.set('sauron_session', routes._create_session())
        assert client.put('/admin/api/documents/d', data={'acl_groups': 'engineering'}).status_code == 200
        assert len(vectors.search([1., 0., 0.], ['engineering'])) == 1
        assert vectors.search([1., 0., 0.], ['old']) == []
        assert client.put('/admin/api/documents/missing', data={'acl_groups': 'engineering'}).status_code == 404
        # If publication of new permissions fails, the index remains narrowed.
        actual = vectors.synchronize_document_acl
        def fail_grant(doc_id, groups):
            if groups == ['new']:
                raise RuntimeError('Simulated index write failure')
            return actual(doc_id, groups)
        monkeypatch.setattr(vectors, 'synchronize_document_acl', fail_grant)
        assert client.put('/admin/api/documents/d', data={'acl_groups': 'new'}).status_code == 503
        assert vectors.search([1., 0., 0.], ['engineering']) == []
        # The same reconciliation used at startup restores the catalog's grant.
        doc = asyncio.run(metadata.get_document('d'))
        assert actual(doc.doc_id, doc.acl_groups)
        assert len(vectors.search([1., 0., 0.], ['new'])) == 1
    asyncio.run(metadata.engine.dispose())
