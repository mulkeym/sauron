import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from src.main import create_app
from src.auth.jwt import create_token

FIXTURES = Path(__file__).parent.parent.parent / "test_fixtures"

@pytest.fixture
def auth_headers():
    token = create_token(username="mike", groups=["finance"])
    return {"Authorization": f"Bearer {token}", "X-API-Key": "test-key-1"}

@pytest.fixture
def client():
    return TestClient(create_app())

def test_ingest_document(client, auth_headers):
    mock_result = MagicMock()
    mock_result.doc_id = "doc-123"
    mock_result.filename = "sample.pdf"
    mock_result.doc_type = "pdf"
    mock_result.chunk_count = 3
    with patch("src.api.routes_ingest.ingest_document", new_callable=AsyncMock, return_value=mock_result):
        with patch("src.api.routes_ingest.get_vector_store", return_value=MagicMock()):
            with patch("src.api.routes_ingest.get_metadata_store", return_value=AsyncMock()):
                with open(FIXTURES / "sample.pdf", "rb") as f:
                    resp = client.post("/api/v1/ingest", files={"file": ("sample.pdf", f, "application/pdf")}, data={"acl_groups": '["finance"]'}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["doc_id"] == "doc-123"

def test_ingest_requires_auth(client):
    resp = client.post("/api/v1/ingest", files={"file": ("test.pdf", b"content", "application/pdf")})
    assert resp.status_code in (401, 403)

def test_list_documents(client, auth_headers):
    mock_doc = MagicMock()
    mock_doc.doc_id = "d1"
    mock_doc.filename = "a.pdf"
    mock_doc.doc_type = "pdf"
    mock_doc.category = ""
    mock_doc.acl_groups = ["finance"]
    mock_doc.chunk_count = 5
    mock_doc.dataset_id = 0
    mock_doc.summary = "A short guide to onboarding edge devices."
    with patch("src.api.routes_ingest.get_metadata_store") as mock_get_store:
        mock_store = AsyncMock()
        mock_store.list_documents.return_value = [mock_doc]
        mock_get_store.return_value = mock_store
        resp = client.get("/api/v1/documents", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["summary"] == "A short guide to onboarding edge devices."


def test_delete_document_removes_all_artifacts(client, auth_headers):
    doc = MagicMock(doc_id="d1", filename="a.pdf", acl_groups=["finance"])
    remaining = MagicMock(doc_id="d2")
    store = AsyncMock()
    store.get_document.return_value = doc
    store.list_documents.return_value = [remaining]
    vector_store = MagicMock()
    schema_registry = MagicMock()

    with patch("src.api.routes_ingest.get_metadata_store", return_value=store), \
         patch("src.api.routes_ingest.get_vector_store", return_value=vector_store), \
         patch("src.api.routes_ingest.get_schema_registry", return_value=schema_registry), \
         patch("src.ingestion.tabular_ingest.cleanup_spreadsheet_tables", new_callable=AsyncMock) as cleanup, \
         patch("src.knowledge.graph_rag.reconcile_lightrag_with_metadata", new_callable=AsyncMock) as reconcile:
        resp = client.delete("/api/v1/documents/d1", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json() == {"doc_id": "d1", "filename": "a.pdf", "status": "deleted"}
    store.delete_document.assert_awaited_once_with("d1")
    store.delete_entities_for_doc.assert_awaited_once_with("d1")
    vector_store.delete_by_doc_id.assert_called_once_with("d1")
    cleanup.assert_awaited_once_with("d1", store, schema_registry)
    reconcile.assert_awaited_once_with({"d2"})


def test_delete_document_rejects_inaccessible_document(client, auth_headers):
    store = AsyncMock()
    store.get_document.return_value = MagicMock(
        doc_id="d1", filename="secret.pdf", acl_groups=["executives"]
    )
    with patch("src.api.routes_ingest.get_metadata_store", return_value=store):
        resp = client.delete("/api/v1/documents/d1", headers=auth_headers)

    assert resp.status_code == 403
    store.delete_document.assert_not_awaited()


def test_delete_document_returns_404(client, auth_headers):
    store = AsyncMock()
    store.get_document.return_value = None
    with patch("src.api.routes_ingest.get_metadata_store", return_value=store):
        resp = client.delete("/api/v1/documents/missing", headers=auth_headers)

    assert resp.status_code == 404
    store.delete_document.assert_not_awaited()


def test_delete_last_document_purges_knowledge_graph(client, auth_headers):
    doc = MagicMock(doc_id="d1", filename="last.pdf", acl_groups=["ALL"])
    store = AsyncMock()
    store.get_document.return_value = doc
    store.list_documents.return_value = []

    with patch("src.api.routes_ingest.get_metadata_store", return_value=store), \
         patch("src.api.routes_ingest.get_vector_store", return_value=MagicMock()), \
         patch("src.ingestion.tabular_ingest.cleanup_spreadsheet_tables", new_callable=AsyncMock), \
         patch("src.knowledge.graph_rag.hard_purge_lightrag", new_callable=AsyncMock) as purge:
        resp = client.delete("/api/v1/documents/d1", headers=auth_headers)

    assert resp.status_code == 200
    purge.assert_awaited_once_with(reason="all metadata documents deleted")
