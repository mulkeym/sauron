from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.auth.jwt import create_token
from src.config import settings
from src.auth.http import EndpointAuthenticationMiddleware
from src.db.schema_registry import SchemaRegistry
from src.mcp.agent_registry import AgentRegistry
from src.mcp.http import add_mcp_http_route, create_mcp_http_app
from src.mcp.server import create_mcp_server


def _app(metadata_store=None):
    metadata_store = metadata_store or AsyncMock()
    server = create_mcp_server(
        vector_store=MagicMock(),
        schema_registry=SchemaRegistry(),
        metadata_store=metadata_store,
        agent_registry=AgentRegistry(),
    )
    mcp_app = create_mcp_http_app(server)
    app = FastAPI(lifespan=mcp_app.lifespan)
    app.add_middleware(EndpointAuthenticationMiddleware)

    @app.get("/admin/")
    async def admin_page():
        return {"admin": True}

    add_mcp_http_route(app, mcp_app)
    return app


def _headers(groups=None):
    return {
        "X-API-Key": "test-key-1",
        "Authorization": "Bearer " + create_token("mike", groups or ["finance"]),
        "Accept": "application/json, text/event-stream",
    }


def _request(method, params=None, request_id=1):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params or {},
    }


def test_native_mcp_rejects_missing_application_key():
    with TestClient(_app()) as client:
        response = client.post("/mcp", json=_request("tools/list"))
    assert response.status_code == 403


def test_native_mcp_does_not_intercept_admin_redirect_or_unknown_routes():
    from src.admin.routes import _create_session, _active_sessions
    token = _create_session()
    with TestClient(_app(), follow_redirects=False) as client:
        client.cookies.set("sauron_session", token)
        admin = client.get("/admin")
        unknown = client.get("/not-an-mcp-route", headers=_headers())
    _active_sessions.discard(token)
    assert admin.status_code == 307
    assert admin.headers["location"].endswith("/admin/")
    assert unknown.status_code == 404


def test_native_mcp_initializes_on_shared_app_path():
    payload = _request(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "1"},
        },
    )
    with TestClient(_app()) as client:
        response = client.post("/mcp", json=payload, headers=_headers())
    assert response.status_code == 200
    assert response.json()["result"]["serverInfo"]["name"] == "sauron"


def test_tool_call_uses_authenticated_groups_not_all():
    metadata_store = AsyncMock()
    metadata_store.list_documents.return_value = []
    payload = _request(
        "tools/call",
        {"name": "tool_list_documents", "arguments": {}},
    )
    with TestClient(_app(metadata_store)) as client:
        response = client.post(
            "/mcp", json=payload, headers=_headers(["finance", "engineering"])
        )
    assert response.status_code == 200
    assert response.json()["result"]["isError"] is False
    metadata_store.list_documents.assert_awaited_once_with(["finance", "engineering"])


def test_openwebui_header_identity_and_forwarded_groups():
    metadata_store = AsyncMock()
    metadata_store.list_documents.return_value = []
    headers = {
        "X-API-Key": "test-key-1",
        "X-Sauron-Username": "user@example.test",
        "X-Sauron-User-Groups": "clinical,finance",
        "Accept": "application/json, text/event-stream",
    }
    payload = _request(
        "tools/call", {"name": "tool_list_documents", "arguments": {}}
    )
    with TestClient(_app(metadata_store)) as client:
        response = client.post("/mcp", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.json()["result"]["isError"] is False
    metadata_store.list_documents.assert_awaited_once_with(["clinical", "finance"])


def test_openwebui_initialize_without_jwt():
    payload = _request(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "1"},
        },
    )
    headers = {
        "X-API-Key": "test-key-1",
        "X-Sauron-Username": "user@example.test",
        "X-Sauron-User-Groups": "finance",
        "Accept": "application/json, text/event-stream",
    }
    with TestClient(_app()) as client:
        response = client.post("/mcp", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.json()["result"]["serverInfo"]["name"] == "sauron"


def test_sync_tool_retains_authenticated_request_context():
    payload = _request(
        "tools/call", {"name": "tool_list_sources", "arguments": {}}
    )
    with patch("src.mcp.server.list_sources", return_value=[]) as list_sources:
        with TestClient(_app()) as client:
            response = client.post(
                "/mcp", json=payload, headers=_headers(["finance"])
            )
    assert response.status_code == 200
    assert response.json()["result"]["isError"] is False
    assert list_sources.call_args.kwargs["user_groups"] == ["finance"]


@pytest.mark.parametrize("groups,expected", [
    ("finance, ALL, engineering,finance", ["finance", "engineering"]),
    ("", []),
])
def test_trusted_openwebui_headers_initialize_and_scope_tool_calls(monkeypatch, groups, expected):
    monkeypatch.setattr(settings, "mcp_openwebui_trust_headers", True)
    monkeypatch.setattr(settings, "mcp_openwebui_jwt_secret", "")
    monkeypatch.setattr(settings, "mcp_openwebui_allow_all_group", False)
    metadata_store = AsyncMock()
    metadata_store.list_documents.return_value = []
    headers = {
        "X-API-Key": "test-key-1",
        "X-OpenWebUI-User-Name": "Mike",
        "X-Sauron-User-Groups": groups,
        "Accept": "application/json, text/event-stream",
    }
    with TestClient(_app(metadata_store)) as client:
        initialized = client.post("/mcp", headers=headers, json=_request("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "openwebui-test", "version": "1"},
        }))
        response = client.post("/mcp", headers=headers, json=_request("tools/call", {
            "name": "tool_list_documents", "arguments": {},
        }))
        rejected = client.post("/mcp", headers={"X-API-Key": "test-key-1"},
                               json=_request("tools/list"))
    assert initialized.status_code == 200
    assert response.status_code == 200
    assert response.json()["result"]["isError"] is False
    assert rejected.status_code == 401
    metadata_store.list_documents.assert_awaited_once_with(expected)
