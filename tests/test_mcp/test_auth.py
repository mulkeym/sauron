# tests/test_mcp/test_auth.py
import pytest

from src.mcp.auth import extract_mcp_context
from src.auth.jwt import create_token
from src.config import settings


def test_extract_context_from_headers():
    token = create_token(username="mike", groups=["finance"])
    headers = {"authorization": f"Bearer {token}", "x-api-key": "test-key-1"}
    ctx = extract_mcp_context(headers)
    assert ctx.username == "mike"
    assert ctx.groups == ["finance"]
    assert ctx.api_key == "test-key-1"
    assert ctx.identity_source == "sauron-jwt"


def test_extract_context_missing_identity():
    headers = {"x-api-key": "test-key-1"}
    with pytest.raises(ValueError, match="Missing user identity"):
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


def test_extract_openwebui_header_identity():
    ctx = extract_mcp_context(
        {
            "X-API-Key": "test-key-1",
            "X-Sauron-Username": "mike@example.test",
            "X-Sauron-User-Groups": "finance, ALL, engineering,finance",
        }
    )
    assert ctx.username == "mike@example.test"
    assert ctx.agent_id == "mike@example.test"
    assert ctx.groups == ["finance", "engineering"]
    assert ctx.identity_source == "openwebui-headers"
    assert ctx.api_key == "test-key-1"


def test_empty_username_header_is_missing_identity():
    with pytest.raises(ValueError, match="Missing user identity"):
        extract_mcp_context(
            {
                "X-API-Key": "test-key-1",
                "X-Sauron-Username": "  ",
                "X-Sauron-User-Groups": "finance",
            }
        )


def test_openwebui_jwt_is_ignored_without_username_header():
    with pytest.raises(ValueError, match="Missing user identity"):
        extract_mcp_context(
            {
                "X-API-Key": "test-key-1",
                "X-OpenWebUI-User-Jwt": "not-a-token",
                "X-Sauron-User-Groups": "finance",
            }
        )


def test_invalid_bearer_does_not_fall_through_to_headers():
    with pytest.raises(ValueError, match="Invalid token"):
        extract_mcp_context(
            {
                "X-API-Key": "test-key-1",
                "Authorization": "Bearer bad-token",
                "X-Sauron-Username": "mike@example.test",
                "X-Sauron-User-Groups": "finance",
            }
        )


def test_username_header_name_is_configurable(monkeypatch):
    monkeypatch.setattr(settings, "mcp_openwebui_username_header", "X-User")
    ctx = extract_mcp_context(
        {
            "X-API-Key": "test-key-1",
            "X-User": "carol",
            "X-Sauron-User-Groups": "contracts",
        }
    )
    assert ctx.username == "carol"
    assert ctx.groups == ["contracts"]
