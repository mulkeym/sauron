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


def test_apply_settings_update_ssl_verify_ignore_checkbox():
    # Models section: hidden ssl_verify=true + optional checkbox value=false (last wins).
    settings.ssl_verify = True
    _apply_settings_update({"ssl_verify": "false"})
    assert settings.ssl_verify is False
    _apply_settings_update({"ssl_verify": "true"})
    assert settings.ssl_verify is True
    # Simulates both form fields when ignore-errors is checked (true then false).
    class _Form:
        def __init__(self, data):
            self._data = data
        def __contains__(self, name):
            return name in self._data
        def getlist(self, name):
            return self._data[name]
    _apply_settings_update(_Form({"ssl_verify": ["true", "false"]}))
    assert settings.ssl_verify is False


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
    from unittest.mock import patch, AsyncMock
    c = _client()
    store = AsyncMock()
    store.list_acl_groups.return_value = []
    store.list_personas.return_value = []
    store.document_acl_group_usage.return_value = {}
    store.discover_orphan_acl_groups.return_value = []
    store.uncovered_document_groups.return_value = []
    store.get_acl_group_names.return_value = []
    for key in ["security", "models", "retrieval", "system", "maintenance"]:
        with patch("src.admin.routes._is_authenticated", return_value=True), \
             patch("src.admin.routes.get_metadata_store", return_value=store):
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


def test_top_nav_drops_moved_links():
    from unittest.mock import patch, AsyncMock
    c = _client()
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store") as mg:
        store = AsyncMock()
        store.list_documents.return_value = []
        store.list_categories.return_value = []
        store.list_proposals.return_value = []
        store.list_recent_query_activity.return_value = []
        mg.return_value = store
        resp = c.get("/admin/")  # dashboard extends base.html (no settings sub-nav)
    assert resp.status_code == 200
    nav = resp.text.split("<nav>")[1].split("</nav>")[0]
    for gone in ['/admin/categories', '/admin/proposals', '/admin/connectors', '/admin/hints', '/admin/audit']:
        assert gone not in nav, gone
    assert '/admin/settings' in nav  # Settings stays


def test_apply_settings_update_partial_other_direction():
    # Symmetric to the Models test: posting SECURITY fields must leave Models/System untouched.
    settings.admin_username = "OLD_ADMIN"
    settings.vllm_model_name = "MODEL_KEEP"
    settings.mcp_port = 9999
    _apply_settings_update({"admin_username": "NEW_ADMIN"})
    assert settings.admin_username == "NEW_ADMIN"     # submitted -> updated
    assert settings.vllm_model_name == "MODEL_KEEP"    # absent -> untouched
    assert settings.mcp_port == 9999                   # absent -> untouched


def test_categories_page_renders_in_shell():
    from unittest.mock import patch, AsyncMock
    c = _client()
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store") as mg:
        store = AsyncMock()
        store.list_categories.return_value = []
        mg.return_value = store
        resp = c.get("/admin/categories")
    assert resp.status_code == 200
    assert 'href="/admin/settings/security"' in resp.text   # rendered inside the settings shell
    assert 'class="subnav-item active"' in resp.text         # an item is active (Categories)
