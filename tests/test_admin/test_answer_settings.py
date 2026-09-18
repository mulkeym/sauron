import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.admin import routes
from src.admin import settings_catalog as catalog
from src.config import settings, Settings, _load_persisted_settings


@pytest.fixture
def client(monkeypatch, tmp_path):
    original = settings.model_dump()
    monkeypatch.setattr(catalog, "SETTINGS_PATH", tmp_path / "settings.json")
    app = FastAPI()
    app.include_router(routes.router)
    c = TestClient(app)
    yield c
    for k, v in original.items():
        setattr(settings, k, v)


def login(client):
    client.cookies.set("sauron_session", routes._create_session())


def test_admin_api_rejects_unauthenticated_read_and_write(client, monkeypatch):
    called = AsyncMock()
    monkeypatch.setattr(routes, "get_metadata_store", lambda: called)
    assert client.post("/admin/api/settings", data={"query_cache_mode": "semantic"}).status_code == 401
    assert client.get("/admin/api/settings/cache-stats").status_code == 401
    assert not catalog.SETTINGS_PATH.exists()
    assert client.post("/admin/api/proposals/1/approve").status_code == 401
    called.approve_proposal.assert_not_called()


def test_admin_api_rejects_cross_origin(client):
    login(client)
    response = client.post("/admin/api/settings", data={"query_cache_mode": "exact"}, headers={"Origin": "https://untrusted.test"})
    assert response.status_code == 403
    assert not catalog.SETTINGS_PATH.exists()


def test_all_settings_render_labeled_controls_without_secrets(client):
    login(client)
    settings.mcp_openwebui_jwt_secret = "do-not-display-this-value"
    response = client.get("/admin/settings/advanced")
    assert response.status_code == 200
    assert "do-not-display-this-value" not in response.text
    for name in Settings.model_fields:
        assert f'name="{name}"' in response.text, name
        assert f'for="{name}"' in response.text, name


def test_valid_settings_persist_and_survive_reload(client, tmp_path, monkeypatch):
    login(client)
    response = client.post("/admin/api/settings", data={
        "query_cache_mode": "exact", "query_cache_ttl_seconds": "900",
        "answer_domain_instructions": "Ask which firmware applies.",
    })
    assert response.status_code == 200
    assert settings.query_cache_mode == "exact"
    saved = json.loads(catalog.SETTINGS_PATH.read_text())
    assert saved["query_cache_ttl_seconds"] == 900
    assert saved["answer_domain_instructions"] == "Ask which firmware applies."
    assert catalog.SETTINGS_PATH.stat().st_mode & 0o077 == 0
    from src.agent.synthesizer import get_system_prompt
    assert "Ask which firmware applies." in get_system_prompt()
    assert "Ask which firmware applies." in client.get("/admin/settings/answers").text
    (tmp_path / "data").mkdir()
    (tmp_path / "data/settings.json").write_text(json.dumps(saved))
    monkeypatch.chdir(tmp_path)
    loaded = _load_persisted_settings(Settings(_env_file=None))
    assert loaded.query_cache_mode == "exact"
    assert loaded.answer_domain_instructions == "Ask which firmware applies."


def test_invalid_partial_update_changes_nothing(client):
    login(client)
    before = settings.model_dump()
    response = client.post("/admin/api/settings", data={"answer_domain_instructions": "Must not apply", "query_cache_ttl_seconds": "-5"})
    assert response.status_code == 422
    assert settings.model_dump() == before
    assert not catalog.SETTINGS_PATH.exists()


def test_restart_fields_are_saved_without_switching_live_datastores(client):
    login(client)
    old = settings.lancedb_path
    response = client.post("/admin/api/settings", data={"lancedb_path": "/new/data/location"})
    assert response.status_code == 200 and "Restart" in response.text
    assert settings.lancedb_path == old
    assert json.loads(catalog.SETTINGS_PATH.read_text())["lancedb_path"] == "/new/data/location"
    response = client.get("/admin/settings/advanced")
    assert "change pending" in response.text and "/new/data/location" in response.text
    client.post("/admin/api/settings", data={"query_cache_mode": "off"})
    assert json.loads(catalog.SETTINGS_PATH.read_text())["lancedb_path"] == "/new/data/location"


def test_blank_secrets_preserve_existing_values(client):
    login(client)
    settings.mcp_openwebui_jwt_secret = "existing-value"
    client.post("/admin/api/settings", data={"keep_blank_secrets": "true", "mcp_openwebui_jwt_secret": ""})
    assert settings.mcp_openwebui_jwt_secret == "existing-value"


def test_trusted_header_settings_save_apply_and_reload(client, tmp_path, monkeypatch):
    from src.mcp.auth import extract_mcp_context, MCPAuthenticationError

    login(client)
    settings.mcp_openwebui_jwt_secret = ""
    headers = {"X-API-Key": "test-key-1", "X-Team-User": "alice", "X-Team-Groups": "team"}
    response = client.post("/admin/api/settings", data={
        "mcp_openwebui_trust_headers": "true",
        "mcp_openwebui_username_header": "X-Team-User",
        "mcp_openwebui_groups_header": "X-Team-Groups",
    })
    assert response.status_code == 200
    assert extract_mcp_context(headers).groups == ["team"]
    page = client.get("/admin/settings/advanced").text
    assert "Trust OpenWebUI user headers" in page
    assert "its holder can assert usernames and groups" in page
    assert 'aria-describedby="mcp_openwebui_trust_headers_help"' in page
    saved = catalog.SETTINGS_PATH.read_text()
    (tmp_path / "data").mkdir()
    (tmp_path / "data/settings.json").write_text(saved)
    monkeypatch.chdir(tmp_path)
    loaded = _load_persisted_settings(Settings(_env_file=None))
    assert loaded.mcp_openwebui_trust_headers is True
    assert loaded.mcp_openwebui_username_header == "X-Team-User"
    client.post("/admin/api/settings", data={"mcp_openwebui_trust_headers": "false"})
    with pytest.raises(MCPAuthenticationError):
        extract_mcp_context(headers)


def test_invalid_username_header_cannot_be_saved(client):
    login(client)
    before = settings.model_dump()
    response = client.post("/admin/api/settings", data={
        "mcp_openwebui_trust_headers": "true", "mcp_openwebui_username_header": "not a header",
    })
    assert response.status_code == 422
    assert settings.model_dump() == before
    assert not catalog.SETTINGS_PATH.exists()
