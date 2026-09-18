from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import routes_openai_compat as compat
from src.auth.jwt import create_token
from src.config import settings
from src.generation.rag_chain import RAGResponse


def test_chat_application_key_cannot_grant_all_groups(monkeypatch):
    app = FastAPI()
    app.include_router(compat.router)
    client = TestClient(app)
    monkeypatch.setattr(settings, "api_keys", "application-key")
    run = AsyncMock(return_value=RAGResponse(answer="answer", citations=[]))
    monkeypatch.setattr(compat, "agent_query", run)
    request = {"messages": [{"role": "user", "content": "deploy"}]}
    for headers in ({"X-API-Key": "application-key"}, {"Authorization": "Bearer application-key"}):
        assert client.post("/v1/chat/completions", json=request, headers=headers).status_code in (401, 403)
    run.assert_not_awaited()
    jwt = create_token("alice", ["team"])
    response = client.post("/v1/chat/completions", json=request,
        headers={"Authorization": "Bearer " + jwt, "X-API-Key": "application-key"})
    assert response.status_code == 200
    assert run.await_args.kwargs["user_groups"] == ["team"]


def test_chat_supports_the_same_trusted_openwebui_headers(monkeypatch):
    app = FastAPI()
    app.include_router(compat.router)
    client = TestClient(app)
    monkeypatch.setattr(settings, "api_keys", "application-key")
    monkeypatch.setattr(settings, "mcp_openwebui_trust_headers", True)
    monkeypatch.setattr(settings, "mcp_openwebui_allow_all_group", False)
    run = AsyncMock(return_value=RAGResponse(answer="answer", citations=[]))
    monkeypatch.setattr(compat, "agent_query", run)
    response = client.post("/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "deploy"}]}, headers={
            "X-API-Key": "application-key", "X-OpenWebUI-User-Name": "Alice",
            "X-Sauron-User-Groups": "team,ALL",
        })
    assert response.status_code == 200
    assert run.await_args.kwargs["user_groups"] == ["team"]
