"""Exercise activity persistence alongside the local evidence/auth additions."""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.admin import routes as admin
from src.agent import profile_store
from src.api import routes_ingest, routes_openai_compat as compat, routes_query
from src.auth.http import EndpointAuthenticationMiddleware
from src.auth.jwt import create_token
from src.config import settings
from src.db.metadata import MetadataStore
from src.generation.rag_chain import RAGResponse
from src.retrieval.models import Citation


def test_queries_keep_evidence_and_reappear_on_dashboard_after_restart(tmp_path, monkeypatch):
    store = MetadataStore(f"sqlite+aiosqlite:///{tmp_path / 'metadata.db'}")
    monkeypatch.setattr(settings, "mcp_openwebui_trust_headers", True)
    monkeypatch.setattr(settings, "mcp_openwebui_username_header", "X-Sauron-Username")
    monkeypatch.setattr(profile_store, "PROFILE_PATH", tmp_path / "profiles.sqlite3")
    vector_store = MagicMock()
    vector_store.table.count_rows.return_value = 0
    citation = Citation(
        doc_id="runbook", filename="runbook.pdf", doc_type="pdf", chunk_index=3,
        snippet="Verify controller connectivity before deployment.", relevance=0.9,
        evidence_id="E123456789012", source_url="https://docs.test/runbook",
        source_locator="page 4", page=4, start_char=20, end_char=70,
    )
    result = RAGResponse(
        answer="Verify controller connectivity [E123456789012].", citations=[citation],
        query_type="lookup", warnings=["Version-specific steps were not found."],
    )
    run = AsyncMock(return_value=result)
    for module in (routes_ingest, routes_query, compat, admin):
        monkeypatch.setattr(module, "get_metadata_store", lambda: store)
        monkeypatch.setattr(module, "get_vector_store", lambda: vector_store)
    monkeypatch.setattr(routes_query, "agent_query", run)
    monkeypatch.setattr(compat, "agent_query", run)

    @asynccontextmanager
    async def lifespan(app):
        await store.init()
        yield
        await store.engine.dispose()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(EndpointAuthenticationMiddleware)
    app.include_router(routes_query.router)
    app.include_router(compat.router)
    app.include_router(admin.router)
    with TestClient(app) as client:
        response = client.post("/api/v1/query", json={"question": "Deploy site A"}, headers={
            "X-API-Key": "test-key-1", "Authorization": "Bearer " + create_token("alice", ["engineering"]),
            "X-Switchyard-Session-Id": "chat-rest",
        })
        assert response.status_code == 200
        assert response.json()["citations"][0]["evidence_id"] == citation.evidence_id
        assert response.json()["citations"][0]["source_url"] == citation.source_url
        assert response.json()["warnings"] == result.warnings
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Deploy site B"}],
        }, headers={
            "X-API-Key": "test-key-1", "X-Sauron-Username": "bob",
            "X-Sauron-User-Groups": "engineering,ALL", "X-Switchyard-Session-Id": "chat-openwebui",
        })
        assert response.status_code == 200
        answer = response.json()["choices"][0]["message"]["content"]
        assert citation.evidence_id not in answer and result.warnings[0] in answer
        assert "[runbook.pdf — page 4](https://docs.test/runbook)" in answer
        assert run.await_args.kwargs["user_groups"] == ["engineering"]
        assert run.await_args.kwargs["session_headers"]["x-switchyard-session-id"] == "chat-openwebui"
        assert run.await_args.kwargs["metadata_store"] is store

    # Open the same SQLite file through a new store/app lifespan.
    store = MetadataStore(f"sqlite+aiosqlite:///{tmp_path / 'metadata.db'}")
    token = admin._create_session()
    try:
        with TestClient(app) as client:
            assert client.get("/admin/", follow_redirects=False).status_code == 302
            client.cookies.set("sauron_session", token)
            page = client.get("/admin/")
            assert page.status_code == 200
            for text in ("Recent queries", "Deploy site A", "Deploy site B", "alice", "bob", "engineering", "lookup"):
                assert text in page.text
            assert client.get("/admin/settings/profiles").status_code == 200
    finally:
        admin._active_sessions.discard(token)
