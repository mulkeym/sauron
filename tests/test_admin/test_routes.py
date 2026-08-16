import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    from src.main import create_app
    return TestClient(create_app())

def _dashboard_store(activity=None, activity_exc=None):
    store = AsyncMock()
    store.list_documents.return_value = [MagicMock() for _ in range(5)]
    store.list_categories.return_value = [MagicMock() for _ in range(3)]
    store.list_proposals.return_value = [MagicMock(), MagicMock()]
    if activity_exc is not None:
        store.list_recent_query_activity.side_effect = activity_exc
    else:
        store.list_recent_query_activity.return_value = activity if activity is not None else []
    return store


def test_dashboard_loads(client):
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store", return_value=_dashboard_store()):
        resp = client.get("/admin/")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text
    assert "Recent queries" in resp.text
    assert "No queries recorded yet." in resp.text


def test_dashboard_shows_activity_row(client):
    from datetime import datetime, timezone
    row = MagicMock()
    row.created_at = datetime(2026, 8, 16, 14, 5, tzinfo=timezone.utc)
    row.source = "mcp"
    row.tool = "ask"
    row.strategy = "lookup"
    row.username = "mike"
    row.user_groups = ["finance"]
    row.query_text = "What is the PTO policy?"
    row.duration_seconds = 3.2
    row.status = "ok"
    row.cache_hit = False
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store", return_value=_dashboard_store([row])):
        resp = client.get("/admin/")
    assert resp.status_code == 200
    assert "MCP · ask" in resp.text
    assert "mike" in resp.text
    assert "What is the PTO policy?" in resp.text
    assert "lookup" in resp.text
    assert "ok" in resp.text


def test_dashboard_activity_read_failure_keeps_stat_cards(client):
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store", return_value=_dashboard_store(activity_exc=RuntimeError("db"))):
        resp = client.get("/admin/")
    assert resp.status_code == 200
    assert "Documents" in resp.text
    assert "Unable to load recent queries." in resp.text
    assert "No queries recorded yet." not in resp.text

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


def test_playground_query_records_activity(client):
    from src.generation.rag_chain import RAGResponse
    from src.agent.graph import AgentTrace
    rec = AsyncMock()
    result = RAGResponse(answer="hi", citations=[], query_type="lookup")
    trace = AgentTrace(query_type="lookup", total_time=1.2)
    store = AsyncMock()
    store.resolve_play_user_groups.return_value = ["finance"]
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store", return_value=store), \
         patch("src.admin.routes.get_vector_store", return_value=MagicMock()), \
         patch("src.admin.routes.get_schema_registry", return_value=MagicMock()), \
         patch("src.agent.graph.run_agent_with_trace", new_callable=AsyncMock, return_value=(result, trace)), \
         patch("src.audit.activity.record_query_activity", rec):
        resp = client.post(
            "/admin/api/playground/query",
            data={"question": "What is PTO?", "play_user": "mike"},
        )
    assert resp.status_code == 200
    rec.assert_awaited()
    kwargs = rec.await_args.kwargs
    assert kwargs["source"] == "playground"
    assert kwargs["tool"] == "playground"
    assert kwargs["username"] == "mike"
    assert kwargs["user_groups"] == ["finance"]
    assert kwargs["query_text"] == "What is PTO?"
    assert kwargs["strategy"] == "lookup"
    assert kwargs["status"] == "ok"
