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


def test_download_secret_is_masked_preserved_and_explicitly_cleared(client):
    login(client)
    secret = "download-admin-secret-that-must-not-be-returned"
    response = client.post("/admin/api/settings", data={"source_download_jwt_secret": secret})
    assert response.status_code == 200
    page = client.get("/admin/settings/advanced").text
    assert secret not in page
    assert 'type="password"' in page
    assert "Original document downloads" in page and "Technical answers" in page
    assert client.post("/admin/api/settings", data={
        "source_download_jwt_secret": "", "keep_blank_secrets": "true",
    }).status_code == 200
    assert settings.source_download_jwt_secret == secret
    assert json.loads(catalog.SETTINGS_PATH.read_text())["source_download_jwt_secret"] == secret
    assert client.post("/admin/api/settings", data={
        "clear_source_download_jwt_secret": "true",
    }).status_code == 200
    assert settings.source_download_jwt_secret == ""


def test_answer_thinking_admin_controls_persist_without_changing_other_settings(client):
    login(client)
    original = settings.vllm_model_name
    html = client.get('/admin/settings/models').text
    assert 'name="llm_answer_thinking"' in html and 'Check thinking support' in html
    assert client.post('/admin/api/settings', data={'llm_answer_thinking':'enabled','llm_reasoning_adapter':'auto'}).status_code == 200
    saved = json.loads(catalog.SETTINGS_PATH.read_text())
    assert saved['llm_answer_thinking'] == 'enabled'
    assert settings.llm_answer_thinking == 'enabled' and settings.vllm_model_name == original
    assert client.post('/admin/api/settings', data={'llm_answer_thinking':'invented'}).status_code == 422


def test_reasoning_capability_is_authenticated_and_truthful(client, monkeypatch):
    from src.generation import reasoning
    lookup = __import__('unittest.mock', fromlist=['MagicMock']).MagicMock(return_value={
        'supported':True,'default_enabled':False,'mandatory':False,'detail':'Verified metadata.'})
    monkeypatch.setattr(reasoning, 'reasoning_capability', lookup)
    assert client.post('/admin/api/settings/reasoning-capability').status_code == 401
    lookup.assert_not_called()
    login(client)
    response = client.post('/admin/api/settings/reasoning-capability', data={'vllm_model_name':'selected'})
    assert 'Provider default: disabled.' in response.text
    assert lookup.call_args.args[1] == 'selected'


@pytest.mark.parametrize('mode', ['default', 'enabled', 'disabled'])
def test_answer_generation_controls_save_and_reload(client, monkeypatch, tmp_path, mode):
    login(client)
    page = client.get('/admin/settings/models').text
    assert 'Final answer generation' in page
    assert 'name="llm_answer_temperature"' in page
    original_model = settings.vllm_model_name
    assert client.post('/admin/api/settings', data={
        'llm_answer_temperature': '1.0', 'llm_answer_thinking': mode,
    }).status_code == 200
    assert settings.llm_answer_temperature == 1.0
    assert settings.vllm_model_name == original_model
    saved = json.loads(catalog.SETTINGS_PATH.read_text())
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/settings.json').write_text(json.dumps(saved))
    monkeypatch.chdir(tmp_path)
    reloaded = _load_persisted_settings(Settings(_env_file=None))
    assert reloaded.llm_answer_temperature == 1.0
    assert reloaded.llm_answer_thinking == mode


@pytest.mark.parametrize('invalid', ['-0.1', '2.01', 'nan', 'inf', 'bad'])
def test_invalid_temperature_rejects_entire_settings_update(client, invalid):
    login(client)
    before = settings.model_dump()
    assert client.post('/admin/api/settings', data={
        'llm_answer_temperature': invalid, 'llm_answer_thinking': 'disabled',
    }).status_code == 422
    assert settings.model_dump() == before
    assert not catalog.SETTINGS_PATH.exists()


@pytest.mark.parametrize('mode', ['default', 'enabled', 'disabled'])
def test_playground_preview_uses_live_answer_controls(client, monkeypatch, mode):
    from unittest.mock import MagicMock
    from src.generation import llm_client
    from src.agent.synthesizer import EvidencePack
    login(client)
    monkeypatch.setattr(settings, 'llm_answer_temperature', 1.0)
    monkeypatch.setattr(settings, 'llm_answer_thinking', mode)
    monkeypatch.setitem(routes._playground_jobs, 'sampling-test', {
        'stream_ready': True,
        'stream_context': {'context': 'evidence', 'question': 'q', 'evidence_pack': EvidencePack()},
    })
    gen = MagicMock(return_value=iter(['SAURON_STATUS: insufficient_evidence']))
    monkeypatch.setattr(llm_client, 'generate_stream', gen)
    response = client.get('/admin/api/playground/stream/sampling-test')
    assert response.status_code == 200
    assert '"done": true' in response.text
    assert gen.call_args.kwargs['temperature'] == 1.0
    assert gen.call_args.kwargs.get('reasoning_mode', 'default') == mode


def test_local_embedding_controls_live_and_validated(client):
    login(client)
    names = {'embedding_batch_size': 12, 'embedding_cpu_threads': 2,
             'embedding_cpu_interop_threads': 1, 'embedding_worker_memory_mb': 2048,
             'embedding_worker_timeout_seconds': 90, 'embedding_worker_idle_seconds': 120}
    page = client.get('/admin/settings/models')
    assert page.status_code == 200
    for name in names:
        assert f'name="{name}"' in page.text and f'for="{name}"' in page.text
    assert 'embedding-status' in page.text
    response = client.post('/admin/api/settings', data=names)
    assert response.status_code == 200 and 'Restart Sauron' not in response.text
    saved = json.loads(catalog.SETTINGS_PATH.read_text())
    for name, value in names.items():
        assert getattr(settings, name) == saved[name] == value
    before = settings.model_dump()
    for change in [{'embedding_batch_size': 0}, {'embedding_cpu_threads': -1},
                   {'embedding_cpu_interop_threads': 0}, {'embedding_worker_memory_mb': 127}]:
        assert client.post('/admin/api/settings', data=change).status_code == 422
        assert settings.model_dump() == before


def test_embedding_status_requires_admin(client):
    assert client.get('/admin/api/settings/embedding-status').status_code == 401
    login(client)
    response = client.get('/admin/api/settings/embedding-status')
    assert response.status_code == 200 and 'Worker:' in response.text


def test_embedding_status_shows_automatic_memory_recovery(client, monkeypatch):
    from src.ingestion import embedding_isolation
    monkeypatch.setattr(embedding_isolation, 'embedding_worker_status', lambda: {
        'state': 'warm', 'effective_threads': 4, 'effective_interop_threads': 1,
        'last_request': {'passages': 64, 'batch_size': 8, 'requested_batch_size': 32,
                         'memory_retries': 2, 'threads': 4, 'interop': 1,
                         'seconds': 10, 'passages_per_second': 6.4,
                         'queue_seconds': 0, 'reused': False},
    })
    login(client)
    response = client.get('/admin/api/settings/embedding-status')
    assert response.status_code == 200
    assert 'requested batch 32, effective batch 8' in response.text
    assert 'Recovered after 2 memory retry/retries' in response.text
