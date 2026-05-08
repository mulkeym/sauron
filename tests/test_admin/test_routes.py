import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    from src.main import create_app
    return TestClient(create_app())

def test_dashboard_loads(client):
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        store.list_documents.return_value = [MagicMock() for _ in range(5)]
        store.list_categories.return_value = [MagicMock() for _ in range(3)]
        store.list_proposals.return_value = [MagicMock(), MagicMock()]
        mock_get.return_value = store
        resp = client.get("/admin/")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text

def test_documents_page(client):
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        doc = MagicMock()
        doc.doc_id = "d1"
        doc.filename = "test.pdf"
        doc.doc_type = "pdf"
        doc.category = "finance"
        doc.acl_groups = ["finance"]
        doc.chunk_count = 5
        doc.uploaded_by = "mike"
        store.list_documents.return_value = [doc]
        mock_get.return_value = store
        resp = client.get("/admin/documents")
    assert resp.status_code == 200
    assert "test.pdf" in resp.text

def test_proposals_page(client):
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        proposal = MagicMock()
        proposal.id = 1
        proposal.proposed_name = "legal"
        proposal.proposed_description = "Legal docs"
        proposal.proposed_acl_groups = ["legal"]
        proposal.proposed_keywords = ["contract"]
        proposal.proposed_by = "system"
        store.list_proposals.return_value = [proposal]
        mock_get.return_value = store
        resp = client.get("/admin/proposals")
    assert resp.status_code == 200
    assert "legal" in resp.text

def test_approve_proposal(client):
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        store.approve_proposal = AsyncMock()
        mock_get.return_value = store
        resp = client.post("/admin/api/proposals/1/approve")
    assert resp.status_code == 200
    store.approve_proposal.assert_called_once()

def test_reject_proposal(client):
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        store.reject_proposal = AsyncMock()
        mock_get.return_value = store
        resp = client.post("/admin/api/proposals/1/reject")
    assert resp.status_code == 200
    store.reject_proposal.assert_called_once()
