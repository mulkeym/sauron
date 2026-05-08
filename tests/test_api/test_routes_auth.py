import pytest
from fastapi.testclient import TestClient
from src.main import create_app

client = TestClient(create_app())

def test_login_returns_token():
    resp = client.post("/api/v1/auth/token", json={"username": "mike", "password": "test", "groups": ["finance"]})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_missing_username():
    resp = client.post("/api/v1/auth/token", json={"password": "test"})
    assert resp.status_code == 422
