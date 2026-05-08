# tests/test_auth/test_dependencies.py
import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from src.auth.dependencies import require_auth
from src.auth.jwt import create_token
from src.auth.models import UserContext

app = FastAPI()


@app.get("/protected")
async def protected_route(user: UserContext = Depends(require_auth)):
    return {"username": user.username, "groups": user.groups}


client = TestClient(app)


def test_valid_auth():
    token = create_token(username="mike", groups=["finance"])
    resp = client.get(
        "/protected",
        headers={
            "Authorization": f"Bearer {token}",
            "X-API-Key": "test-key-1",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["username"] == "mike"
    assert resp.json()["groups"] == ["finance"]


def test_missing_api_key():
    token = create_token(username="mike", groups=["finance"])
    resp = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


def test_invalid_api_key():
    token = create_token(username="mike", groups=["finance"])
    resp = client.get(
        "/protected",
        headers={
            "Authorization": f"Bearer {token}",
            "X-API-Key": "wrong-key",
        },
    )
    assert resp.status_code == 403


def test_missing_jwt():
    resp = client.get(
        "/protected",
        headers={"X-API-Key": "test-key-1"},
    )
    assert resp.status_code == 401


def test_invalid_jwt():
    resp = client.get(
        "/protected",
        headers={
            "Authorization": "Bearer garbage-token",
            "X-API-Key": "test-key-1",
        },
    )
    assert resp.status_code == 401
