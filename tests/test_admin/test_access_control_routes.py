"""Admin routes for ACL groups and personas (Security settings)."""
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


def _client():
    from src.main import create_app
    return TestClient(create_app())


def _group(name="finance", display_name="Finance", description="Budget", active=True):
    g = MagicMock()
    g.name = name
    g.display_name = display_name
    g.description = description
    g.active = active
    return g


def _persona(name="mike", display_name="Mike", role="Finance Manager",
             groups=None, active=True, sort_order=10):
    p = MagicMock()
    p.name = name
    p.display_name = display_name
    p.role = role
    p.groups = groups if groups is not None else ["finance", "executives"]
    p.active = active
    p.sort_order = sort_order
    return p


def _store():
    store = AsyncMock()
    store.list_acl_groups.return_value = [_group()]
    store.list_personas.return_value = [_persona()]
    store.document_acl_group_usage.return_value = {"finance": 2}
    store.discover_orphan_acl_groups.return_value = []
    store.uncovered_document_groups.return_value = []
    store.get_acl_group_names.return_value = ["finance", "executives"]
    store.get_acl_group.return_value = _group()
    store.get_persona.return_value = _persona()
    store.add_acl_group.return_value = _group(name="clinical", display_name="Clinical")
    store.add_persona.return_value = _persona(name="rita", display_name="Rita", groups=["clinical"])
    store.update_acl_group.return_value = _group(display_name="Finance Team")
    store.update_persona.return_value = _persona(groups=["finance", "clinical"])
    store.set_acl_group_active.return_value = _group(active=False)
    store.set_persona_active.return_value = _persona(active=False)
    store.delete_persona = AsyncMock()
    store.resolve_play_user_groups.return_value = ["finance", "executives"]
    store.list_datasets.return_value = []
    return store


def test_security_page_renders_access_sections():
    c = _client()
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store", return_value=_store()):
        resp = c.get("/admin/settings/security")
    assert resp.status_code == 200
    assert "ACL Groups" in resp.text
    assert "Playground Personas" in resp.text
    assert "finance" in resp.text
    assert "Mike" in resp.text


def test_add_acl_group():
    c = _client()
    store = _store()
    store.get_acl_group.return_value = None  # not existing
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store", return_value=store):
        resp = c.post("/admin/api/acl-groups/add", data={
            "name": "clinical",
            "display_name": "Clinical",
            "description": "Clinical staff",
        })
    assert resp.status_code == 200
    assert "created" in resp.text.lower()
    store.add_acl_group.assert_awaited()


def test_add_persona():
    c = _client()
    store = _store()
    store.get_persona.return_value = None
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store", return_value=store):
        resp = c.post("/admin/api/personas/add", data={
            "name": "rita",
            "display_name": "Rita",
            "role": "Nurse",
            "groups": ["clinical"],
        })
    assert resp.status_code == 200
    assert "created" in resp.text.lower()
    store.add_persona.assert_awaited()


def test_persona_edit_form():
    c = _client()
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store", return_value=_store()):
        resp = c.get("/admin/api/personas/mike/edit")
    assert resp.status_code == 200
    assert 'name="display_name"' in resp.text
    assert 'type="checkbox"' in resp.text


def test_playground_uses_personas():
    c = _client()
    store = _store()
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store", return_value=store):
        resp = c.get("/admin/playground")
    assert resp.status_code == 200
    assert "Mike" in resp.text
    assert 'value="mike"' in resp.text
    assert "Custom groups" in resp.text
