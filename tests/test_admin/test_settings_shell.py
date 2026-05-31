from src.admin.routes import _apply_settings_update
from src.config import settings


def test_apply_settings_update_is_partial():
    settings.admin_username = "ADMIN_KEEP"
    settings.llm_max_context = 250000
    settings.vllm_model_name = "OLD"
    persist = _apply_settings_update({"vllm_model_name": "NEW", "vllm_base_url": "http://x"})
    assert settings.vllm_model_name == "NEW"          # submitted -> updated
    assert settings.admin_username == "ADMIN_KEEP"     # absent -> untouched
    assert settings.llm_max_context == 250000          # absent -> untouched
    assert persist["vllm_model_name"] == "NEW"
    assert persist["admin_username"] == "ADMIN_KEEP"
    assert persist["llm_max_context"] == 250000


def test_apply_settings_update_bool_roundtrip():
    settings.feedback_enabled = True
    _apply_settings_update({"feedback_enabled": "false"})   # unchecked box posts hidden "false"
    assert settings.feedback_enabled is False
    _apply_settings_update({"feedback_enabled": "true"})
    assert settings.feedback_enabled is True


def test_apply_settings_update_keeps_blank_credential():
    settings.admin_password = "secret"
    _apply_settings_update({"admin_password": ""})           # blank submit must not clear it
    assert settings.admin_password == "secret"


def test_apply_settings_update_casts_numbers():
    _apply_settings_update({"llm_max_context": "300000", "feedback_similarity_threshold": "0.7"})
    assert settings.llm_max_context == 300000 and isinstance(settings.llm_max_context, int)
    assert settings.feedback_similarity_threshold == 0.7


def test_apply_settings_update_can_clear_vllm_api_key():
    from src.admin.routes import _apply_settings_update
    from src.config import settings
    settings.vllm_api_key = "sk-old"
    _apply_settings_update({"vllm_api_key": ""})   # blank submit -> cleared (local model)
    assert settings.vllm_api_key == ""


def test_apply_settings_update_absent_bool_unchanged():
    from src.admin.routes import _apply_settings_update
    from src.config import settings
    settings.feedback_enabled = True
    _apply_settings_update({})   # no checkbox AND no hidden field -> field absent -> keep
    assert settings.feedback_enabled is True


def test_apply_settings_update_blank_numeric_kept():
    from src.admin.routes import _apply_settings_update
    from src.config import settings
    settings.llm_max_context = 250000
    _apply_settings_update({"llm_max_context": ""})  # blank numeric -> keep current (no int("") crash)
    assert settings.llm_max_context == 250000


def _client():
    from fastapi.testclient import TestClient
    from src.main import create_app
    return TestClient(create_app())


def test_settings_redirects_to_first_section():
    from unittest.mock import patch
    with patch("src.admin.routes._is_authenticated", return_value=True):
        resp = _client().get("/admin/settings", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"].endswith("/admin/settings/security")


def test_each_config_section_renders_with_active_marker():
    from unittest.mock import patch
    c = _client()
    for key in ["security", "models", "retrieval", "system", "maintenance"]:
        with patch("src.admin.routes._is_authenticated", return_value=True):
            resp = c.get(f"/admin/settings/{key}")
        assert resp.status_code == 200, key
        assert 'class="subnav-item active"' in resp.text, key
        assert f'href="/admin/settings/{key}"' in resp.text, key


def test_management_pages_render_in_shell():
    from unittest.mock import patch, AsyncMock
    c = _client()
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store") as mg:
        store = AsyncMock()
        store.load_all_hints.return_value = []
        store.list_datasets.return_value = []
        mg.return_value = store
        resp = c.get("/admin/hints")
    assert resp.status_code == 200
    # rendered inside the settings shell -> sub-nav (a link to a settings section) is present
    assert 'href="/admin/settings/security"' in resp.text
    # Schema Hints is the active sub-nav item
    assert 'class="subnav-item active"' in resp.text
