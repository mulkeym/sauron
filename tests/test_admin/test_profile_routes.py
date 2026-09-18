from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.admin import routes, profile_routes
from src.agent import profile_store
from src.agent.profiles import active_snapshot


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_store, "PROFILE_PATH", tmp_path / "profiles.sqlite3")
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def login(client):
    client.cookies.set("sauron_session", routes._create_session())


def test_profile_page_and_every_api_require_admin_session(client):
    assert client.get("/admin/settings/profiles", follow_redirects=False).status_code == 302
    for method, path in (
        ("GET", ""), ("POST", ""), ("PUT", "/general/draft"),
        ("POST", "/general/publish"), ("POST", "/general/revisions/1/activate"),
        ("POST", "/general/preview"), ("POST", "/general/prompt"),
    ):
        assert client.request(method, "/admin/api/answer-profiles" + path, json={}).status_code == 401
    assert not profile_store.PROFILE_PATH.exists()


def test_profile_editor_and_cross_origin_protection(client):
    login(client)
    response = client.get("/admin/settings/profiles")
    assert response.status_code == 200
    for text in ("Run draft preview", "Publish saved draft", "Activate selected revision", "Answer instructions", "Routing strategy"):
        assert text in response.text
    response = client.post("/admin/api/answer-profiles/general/publish", json={"expected_version": 0}, headers={"Origin": "https://elsewhere.test"})
    assert response.status_code == 403
    assert not profile_store.PROFILE_PATH.exists()


def test_admin_save_publish_rollback_and_stale_tab(client):
    login(client)
    root = "/admin/api/answer-profiles"
    book = client.get(root).json()
    config = book["profiles"]["general"]["draft"]
    config["instructions"] = "Start with documented prerequisites."
    response = client.put(root + "/general/draft", json={"expected_version": 0, "config": config})
    assert response.status_code == 200
    assert active_snapshot()["config"]["instructions"] != config["instructions"]
    response = client.post(root + "/general/publish", json={"expected_version": 0})
    assert response.status_code == 409
    response = client.post(root + "/general/publish", json={"expected_version": 1})
    assert response.status_code == 200
    assert active_snapshot()["config"]["instructions"] == config["instructions"]
    response = client.post(root + "/general/revisions/1/activate", json={"expected_version": 2})
    assert response.status_code == 200
    assert active_snapshot()["revision"] == 1
    assert response.json()["profiles"]["general"]["draft"] == config
    assert len(response.json()["profiles"]["general"]["revisions"]) == 2


def test_invalid_settings_cannot_change_draft(client):
    login(client)
    before = profile_store.read_book()
    invalid = {**before["profiles"]["general"]["draft"], "strategy": "unknown"}
    response = client.put("/admin/api/answer-profiles/general/draft", json={"expected_version": 0, "config": invalid})
    assert response.status_code == 422
    assert profile_store.read_book() == before


def test_unsaved_preview_and_prompt_leave_store_unchanged(client, monkeypatch):
    login(client)
    before = profile_store.read_book()
    config = {**before["profiles"]["general"]["draft"], "instructions": "Unsaved testing instructions"}
    run = AsyncMock(return_value={"answer": "Test answer", "citations": []})
    monkeypatch.setattr(profile_routes, "run_preview", run)
    response = client.post("/admin/api/answer-profiles/general/preview", json={"config": config, "question": "How do we deploy?", "user_groups": ["engineering"], "dataset_id": 4})
    assert response.status_code == 200
    request = run.call_args.args[1]
    assert request.config.instructions == "Unsaved testing instructions"
    assert request.user_groups == ["engineering"] and request.dataset_id == 4
    response = client.post("/admin/api/answer-profiles/general/prompt", json={"config": config})
    assert response.status_code == 200
    assert "Unsaved testing instructions" in response.json()["system_prompt"]
    assert profile_store.read_book() == before
    assert not profile_store.PROFILE_PATH.exists()


def test_failed_preview_does_not_publish(client, monkeypatch):
    login(client)
    monkeypatch.setattr(profile_routes, "run_preview", AsyncMock(side_effect=RuntimeError("synthetic unavailable model")))
    before = profile_store.read_book()
    response = client.post("/admin/api/answer-profiles/general/preview", json={"config": before["profiles"]["general"]["draft"], "question": "deploy"})
    assert response.status_code == 502
    assert "Preview failed" in response.json()["detail"]
    assert profile_store.read_book() == before


def test_profile_copy_does_not_activate_or_copy_publication_history(client):
    login(client)
    before = profile_store.read_book()
    response = client.post("/admin/api/answer-profiles", json={"expected_version": 0, "config": {**before["profiles"]["deployment"]["draft"], "name": "Branch deployment"}})
    assert response.status_code == 200
    created = response.json()
    assert created["book"]["profiles"][created["profile_id"]]["revisions"] == []
    assert created["book"]["active"] == before["active"]
