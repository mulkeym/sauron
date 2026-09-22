"""Exercise the real application boundary, including the settings-reset exploit."""
import json
import re
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.admin import routes as admin
from src.auth.api_key import ApiKeyContext, hash_api_key
from src.auth.jwt import create_token
from src.config import settings
from src.main import create_app


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(settings, "admin_username", "auth-test-admin")
    monkeypatch.setattr(settings, "admin_password", "auth-test-password")
    monkeypatch.setattr(settings, "api_keys", "test-key-1,test-key-2")
    monkeypatch.setattr(settings, "ssl_verify", True)
    monkeypatch.setattr(admin, "_active_sessions", set())
    return TestClient(create_app(), follow_redirects=False)


def sign_in(client):
    response = client.post("/admin/login", data={
        "username": settings.admin_username,
        "password": settings.admin_password,
    })
    assert response.status_code == 302
    assert client.cookies.get("sauron_session")


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "invalid"}])
def test_every_registered_endpoint_denies_anonymous_requests(client, headers):
    checked = 0
    def endpoints(routes, prefix=""):
        for route in routes:
            # FastAPI 0.141 keeps included routers as lazy branches.
            if hasattr(route, "original_router"):
                yield from endpoints(route.original_router.routes,
                                     prefix + route.include_context.prefix)
            else:
                yield route, prefix + getattr(route, "path", "")

    for route, path in endpoints(client.app.routes):
        if path == "/admin/login" or path == "/admin/static":
            continue
        path = re.sub(r"\{[^}]+\}", "1", path)
        for method in getattr(route, "methods", None) or {"GET", "POST", "DELETE"}:
            response = client.request(method, path, headers=headers)
            if path.startswith("/admin/"):
                expected = 302 if method in {"GET", "HEAD"} and not path.startswith("/admin/api/") else 401
            elif path == "/api/v1/figure-links/1" and method in {"GET", "HEAD"}:
                expected = 404  # The public image route rejects invalid capabilities itself.
            else:
                expected = 403
            assert response.status_code == expected, (method, path, response.status_code)
            checked += 1
    assert checked > 100  # Includes mounted MCP, docs, REST, and all admin actions.


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "test-key-1"}])
def test_password_reset_requires_admin_session_and_has_no_side_effects(client, tmp_path, headers):
    old_password = settings.admin_password
    response = client.post("/admin/api/settings", headers=headers, data={"admin_password": "attacker"})
    assert response.status_code == 401
    assert settings.admin_password == old_password
    assert not (tmp_path / "data/settings.json").exists()


def test_admin_login_save_and_logout(client, tmp_path):
    sign_in(client)
    response = client.post("/admin/api/settings", data={"admin_password": "new-password"})
    assert response.status_code == 200
    saved = json.loads((tmp_path / "data/settings.json").read_text())
    assert saved["admin_password"] == "new-password"
    assert client.get("/admin/logout").status_code == 302
    assert client.post("/admin/api/settings", data={"admin_password": "attacker"}).status_code == 401
    assert settings.admin_password == "new-password"
    assert client.post("/admin/login", data={"username": settings.admin_username, "password": "new-password"}).status_code == 302


def test_wrong_password_and_forged_cookie_are_rejected(client):
    response = client.post("/admin/login", data={"username": settings.admin_username, "password": "wrong"})
    assert "Invalid username or password" in response.text
    assert not client.cookies.get("sauron_session")
    client.cookies.set("sauron_session", "forged")
    assert client.post("/admin/api/settings", data={"admin_password": "attacker"}).status_code == 401


@pytest.mark.parametrize("path", ["/api/health", "/api/v1/auth/token", "/v1/models", "/openapi.json", "/docs", "/redoc", "/unknown", "/admin-other"])
def test_admin_cookie_never_replaces_an_api_key(client, path):
    sign_in(client)
    assert client.get(path).status_code == 403


@pytest.mark.parametrize("path", ["/api/health", "/v1/models", "/openapi.json", "/docs", "/redoc"])
def test_api_key_allows_service_metadata_endpoints(client, path):
    assert client.get(path, headers={"X-API-Key": "test-key-1"}).status_code == 200


def test_token_issuance_requires_application_key(client):
    body = {"username": "mike", "password": "lab-only", "groups": ["finance"]}
    assert client.post("/api/v1/auth/token", json=body).status_code == 403
    response = client.post("/api/v1/auth/token", json=body, headers={"X-API-Key": "test-key-1"})
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_openai_bearer_key_works_but_jwt_alone_does_not(client):
    assert client.get("/v1/models", headers={"Authorization": "Bearer test-key-1"}).status_code == 200
    jwt = create_token("mike", ["finance"])
    headers = {"Authorization": "Bearer " + jwt}
    with patch("src.api.routes_openai_compat.agent_query", new_callable=AsyncMock) as query:
        assert client.post("/v1/chat/completions", headers=headers, json={"messages": [{"role": "user", "content": "secret?"}]}).status_code == 403
        query.assert_not_awaited()
    assert client.get("/v1/models", headers={"X-API-Key": "invalid", "Authorization": "Bearer test-key-1"}).status_code == 403


def test_db_application_key_and_revocation_are_enforced(client, monkeypatch):
    from src.auth import api_key
    key = "db-application-secret"
    cache = {hash_api_key(key): ApiKeyContext(source="db", application_id=1, key_id=1)}
    monkeypatch.setattr(api_key, "_db_key_cache", cache)
    assert client.get("/api/health", headers={"X-API-Key": key}).status_code == 200
    cache.clear()
    assert client.get("/api/health", headers={"X-API-Key": key}).status_code == 403


def test_no_configured_keys_fails_closed(client, monkeypatch):
    from src.auth import api_key
    monkeypatch.setattr(settings, "api_keys", "")
    monkeypatch.setattr(api_key, "_db_key_cache", {})
    for key in ["", "dev-key-1", "test-key-1"]:
        assert client.get("/api/health", headers={"X-API-Key": key}).status_code == 403
    assert client.get("/admin/login").status_code == 200


def test_new_install_has_no_shared_development_key(monkeypatch, tmp_path):
    from src.config import Settings
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("API_KEYS", raising=False)
    assert Settings().api_key_list == []


def test_login_assets_and_only_real_cors_preflight_remain_public(client):
    assert client.get("/admin/login").status_code == 200
    assert client.get("/admin/static/style.css").status_code == 200
    assert client.post("/admin/static/style.css").status_code == 401
    assert client.options("/api/v1/auth/token").status_code == 403
    response = client.options("/api/v1/auth/token", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-API-Key,Content-Type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    # A preflight never authorizes the corresponding application request.
    assert client.post("/api/v1/auth/token", headers={"Origin": "http://localhost:5173"}).status_code == 403


@pytest.mark.parametrize("source", ["https://evil.example", "http://localhost:5173", "null", "http://testserver:9999", "http://testserver:0", "http://testserver:invalid"])
def test_cross_origin_admin_reads_and_writes_are_denied(client, tmp_path, source):
    sign_in(client)
    headers = {"Origin": source}
    assert client.get("/admin/api/backup/list", headers=headers).status_code == 403
    assert client.post("/admin/api/settings", headers=headers, data={"admin_password": "attacker"}).status_code == 403
    assert not (tmp_path / "data/settings.json").exists()


def test_same_origin_admin_save_and_external_navigation(client):
    sign_in(client)
    assert client.post("/admin/api/settings", headers={"Origin": "http://testserver:80"}, data={"admin_password": "new-password"}).status_code == 200
    assert client.get("/admin/login", headers={"Referer": "https://external.example/"}).status_code == 200


def test_new_route_is_protected_without_opt_in(client):
    calls = []

    @client.app.get("/future-api")
    async def future_api():
        calls.append(True)
        return {"ok": True}

    assert client.get("/future-api").status_code == 403
    assert not calls
    assert client.get("/future-api", headers={"X-API-Key": "test-key-1"}).status_code == 200
    assert calls == [True]


def test_websocket_routes_also_require_application_keys(client):
    from fastapi import WebSocket
    from starlette.websockets import WebSocketDisconnect

    @client.app.websocket("/future-socket")
    async def future_socket(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"ok": True})
        await websocket.close()

    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect("/future-socket"):
            pass
    assert error.value.code == 1008
    with client.websocket_connect("/future-socket", headers={"X-API-Key": "test-key-1"}) as socket:
        assert socket.receive_json() == {"ok": True}


def test_mcp_still_requires_user_identity_after_api_key(client):
    payload = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
    assert client.post("/mcp", json=payload).status_code == 403
    assert client.post("/mcp", json=payload, headers={"X-API-Key": "test-key-1"}).status_code == 401


def test_rest_still_requires_user_jwt_after_api_key(client):
    response = client.get("/api/v1/documents", headers={"X-API-Key": "test-key-1"})
    assert response.status_code == 401


def test_root_path_does_not_change_auth_boundaries(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "test-key-1")
    client = TestClient(create_app(), root_path="/sauron", follow_redirects=False)
    assert client.get("/sauron/admin/login").status_code == 200
    response = client.get("/sauron/admin/")
    assert response.status_code == 302
    assert response.headers["location"] == "/sauron/admin/login"
    assert client.post("/sauron/admin/api/settings").status_code == 401
    assert client.get("/sauron/api/health").status_code == 403
