import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from src.main import create_app
from src.auth.jwt import create_token
from src.generation.rag_chain import RAGResponse
from src.retrieval.models import Citation

@pytest.fixture
def auth_headers():
    token = create_token(username="mike", groups=["finance"])
    return {"Authorization": f"Bearer {token}", "X-API-Key": "test-key-1"}

@pytest.fixture
def client():
    return TestClient(create_app())

def test_query_returns_answer_with_citations(client, auth_headers):
    mock_response = RAGResponse(
        answer="Expenses over $500 need approval [1].",
        citations=[Citation(doc_id="doc-1", filename="policy.pdf", doc_type="pdf", chunk_index=0, page=12, snippet="All expenses over $500...", relevance=0.95)],
    )
    with patch("src.api.routes_query.agent_query", new_callable=AsyncMock, return_value=mock_response):
        with patch("src.api.routes_query.get_vector_store", return_value=MagicMock()):
            with patch("src.api.routes_query.get_schema_registry", return_value=MagicMock()):
                resp = client.post("/api/v1/query", json={"question": "What is the expense policy?"}, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["citations"]) == 1
    assert data["citations"][0]["filename"] == "policy.pdf"

def test_query_requires_auth(client):
    resp = client.post("/api/v1/query", json={"question": "test"})
    assert resp.status_code in (401, 403)

def test_query_empty_question(client, auth_headers):
    with patch("src.api.routes_query.agent_query", new_callable=AsyncMock, return_value=RAGResponse(answer="No info found.", citations=[])):
        with patch("src.api.routes_query.get_vector_store", return_value=MagicMock()):
            with patch("src.api.routes_query.get_schema_registry", return_value=MagicMock()):
                resp = client.post("/api/v1/query", json={"question": ""}, headers=auth_headers)
    assert resp.status_code == 200
