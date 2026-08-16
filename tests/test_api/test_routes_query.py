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


def test_query_passes_skip_cache(client, auth_headers):
    mock_response = RAGResponse(answer="fresh", citations=[])
    with patch("src.api.routes_query.agent_query", new_callable=AsyncMock, return_value=mock_response) as mock_aq:
        with patch("src.api.routes_query.get_vector_store", return_value=MagicMock()):
            with patch("src.api.routes_query.get_schema_registry", return_value=MagicMock()):
                with patch("src.api.routes_query.get_metadata_store", return_value=MagicMock()):
                    resp = client.post(
                        "/api/v1/query",
                        json={"question": "expense policy?", "skip_cache": True},
                        headers=auth_headers,
                    )
    assert resp.status_code == 200
    mock_aq.assert_awaited_once()
    assert mock_aq.await_args.kwargs.get("skip_cache") is True
    assert mock_aq.await_args.kwargs.get("agent_id") == "mike"
    assert mock_aq.await_args.kwargs.get("session_headers") is not None


def test_query_forwards_inbound_session_header(client, auth_headers):
    mock_response = RAGResponse(answer="ok", citations=[])
    headers = {**auth_headers, "X-OpenWebUI-Chat-Id": "chat-7"}
    with patch("src.api.routes_query.agent_query", new_callable=AsyncMock, return_value=mock_response) as mock_aq:
        with patch("src.api.routes_query.get_vector_store", return_value=MagicMock()):
            with patch("src.api.routes_query.get_schema_registry", return_value=MagicMock()):
                with patch("src.api.routes_query.get_metadata_store", return_value=MagicMock()):
                    resp = client.post(
                        "/api/v1/query",
                        json={"question": "tell me about sdwan"},
                        headers=headers,
                    )
    assert resp.status_code == 200
    forwarded = mock_aq.await_args.kwargs["session_headers"]
    assert forwarded.get("x-openwebui-chat-id") == "chat-7" or forwarded.get("X-OpenWebUI-Chat-Id") == "chat-7"
