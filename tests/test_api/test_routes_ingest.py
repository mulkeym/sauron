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
    with patch("src.api.routes_ingest.get_metadata_store") as mock_get_store:
        mock_store = AsyncMock()
        mock_store.list_documents.return_value = [mock_doc]
        mock_get_store.return_value = mock_store
        resp = client.get("/api/v1/documents", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 1
