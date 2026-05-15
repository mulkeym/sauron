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

def test_knowledge_graph_filtered_by_app(client):
    """Filtering by app_id returns only entities from that app's documents."""
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        store.list_applications.return_value = []
        mock_get.return_value = store

        with patch("src.admin.routes._load_lightrag_graph") as mock_graph:
            mock_graph.return_value = (
                [{"name": "Acme Corp", "type": "organization"}, {"name": "Bob Smith", "type": "person"}],
                [{"source": "Acme Corp", "target": "Bob Smith", "label": "employs"}],
            )
            with patch("src.knowledge.graph_rag._get_app_allowed_entities") as mock_app_filter:
                mock_app_filter.return_value = {"Acme Corp"}
                resp = client.get("/admin/api/knowledge-graph/filtered?app_id=1")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["entities"]) == 1
    assert data["entities"][0]["name"] == "Acme Corp"
    assert len(data["relationships"]) == 0  # Bob filtered out, so edge is removed


def test_knowledge_graph_filtered_by_app_and_persona(client):
    """When both app_id and groups are set, result is the intersection."""
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        mock_get.return_value = store

        with patch("src.admin.routes._load_lightrag_graph") as mock_graph:
            mock_graph.return_value = (
                [
                    {"name": "Acme Corp", "type": "organization"},
                    {"name": "Bob Smith", "type": "person"},
                    {"name": "Secret Project", "type": "concept"},
                ],
                [],
            )
            with patch("src.knowledge.graph_rag._get_app_allowed_entities") as mock_app:
                mock_app.return_value = {"Acme Corp", "Bob Smith"}  # app has these two
                with patch("src.knowledge.graph_rag._get_acl_allowed_entities") as mock_acl:
                    mock_acl.return_value = {"Acme Corp", "Secret Project"}  # persona sees these two
                    resp = client.get("/admin/api/knowledge-graph/filtered?app_id=1&groups=finance")

    assert resp.status_code == 200
    data = resp.json()
    # Intersection: only "Acme Corp" is in both sets
    assert len(data["entities"]) == 1
    assert data["entities"][0]["name"] == "Acme Corp"
