# tests/test_mcp/test_auth.py
import pytest
import jwt
from datetime import datetime, timedelta, timezone

from src.mcp.auth import extract_mcp_context, MCPContext
from src.auth.jwt import create_token
from src.config import settings

def test_extract_context_from_headers():
    token = create_token(username="mike", groups=["finance"])
    headers = {"authorization": f"Bearer {token}", "x-api-key": "test-key-1"}
    ctx = extract_mcp_context(headers)
    assert ctx.username == "mike"
    assert ctx.groups == ["finance"]
    assert ctx.api_key == "test-key-1"

def test_extract_context_missing_jwt():
    headers = {"x-api-key": "test-key-1"}
    with pytest.raises(ValueError, match="Missing"):
        extract_mcp_context(headers)

def test_extract_context_missing_api_key():
    token = create_token(username="mike", groups=["finance"])
    headers = {"authorization": f"Bearer {token}"}
    with pytest.raises(ValueError, match="API key"):
        extract_mcp_context(headers)

def test_extract_context_invalid_jwt():
    headers = {"authorization": "Bearer bad-token", "x-api-key": "test-key-1"}
    with pytest.raises(ValueError, match="Invalid token"):
        extract_mcp_context(headers)


def test_extract_openwebui_context(monkeypatch):
    secret = "openwebui-test-secret-at-least-32-bytes"
    monkeypatch.setattr(settings, "mcp_openwebui_jwt_secret", secret)
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": "user-123",
            "email": "mike@example.test",
            "name": "Mike",
            "role": "user",
            "iss": "open-webui",
            "iat": now,
            "exp": now + timedelta(minutes=5),
        },
        secret,
        algorithm="HS256",
    )
    ctx = extract_mcp_context(
        {
            "X-API-Key": "test-key-1",
            "X-OpenWebUI-User-Jwt": token,
            "X-Sauron-User-Groups": "finance, ALL, engineering,finance",
        }
    )
    assert ctx.username == "mike@example.test"
    assert ctx.agent_id == "user-123"
    assert ctx.groups == ["finance", "engineering"]
    assert ctx.identity_source == "openwebui-jwt"


def test_openwebui_identity_requires_configured_secret(monkeypatch):
    monkeypatch.setattr(settings, "mcp_openwebui_jwt_secret", "")
    with pytest.raises(ValueError, match="not configured"):
        extract_mcp_context(
            {
                "X-API-Key": "test-key-1",
                "X-OpenWebUI-User-Jwt": "not-a-token",
            }
        )
