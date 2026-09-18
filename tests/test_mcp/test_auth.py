# tests/test_mcp/test_auth.py
import pytest
import jwt
from datetime import datetime, timedelta, timezone

from src.mcp.auth import extract_mcp_context, MCPContext, MCPAuthenticationError
from src.auth.jwt import create_token
from src.config import settings


@pytest.fixture(autouse=True)
def forwarding_defaults(monkeypatch):
    monkeypatch.setattr(settings, "mcp_openwebui_trust_headers", False)
    monkeypatch.setattr(settings, "mcp_openwebui_username_header", "X-OpenWebUI-User-Name")
    monkeypatch.setattr(settings, "mcp_openwebui_groups_header", "X-Sauron-User-Groups")
    monkeypatch.setattr(settings, "mcp_openwebui_allow_all_group", False)


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


@pytest.mark.parametrize("trust_headers", [False, True])
def test_extract_openwebui_context(monkeypatch, trust_headers):
    monkeypatch.setattr(settings, "mcp_openwebui_trust_headers", trust_headers)
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


def _unsigned_headers():
    return {
        "X-API-Key": "test-key-1",
        "X-OpenWebUI-User-Name": " Mike ",
        "X-OpenWebUI-User-Id": "user-123",
        "X-Sauron-User-Groups": "finance, ALL, engineering,finance, ",
    }


def test_unsigned_openwebui_headers_require_opt_in():
    with pytest.raises(MCPAuthenticationError):
        extract_mcp_context(_unsigned_headers())


def test_trusted_openwebui_headers_need_no_jwt_secret(monkeypatch):
    monkeypatch.setattr(settings, "mcp_openwebui_trust_headers", True)
    monkeypatch.setattr(settings, "mcp_openwebui_jwt_secret", "")
    ctx = extract_mcp_context(_unsigned_headers())
    assert ctx.username == "Mike"
    assert ctx.groups == ["finance", "engineering"]
    assert ctx.agent_id == "user-123"
    assert ctx.identity_source == "openwebui-headers"


@pytest.mark.parametrize("key", [None, "", "wrong-key"])
def test_trusted_headers_still_require_valid_api_key(monkeypatch, key):
    monkeypatch.setattr(settings, "mcp_openwebui_trust_headers", True)
    headers = _unsigned_headers()
    if key is None:
        headers.pop("X-API-Key")
    else:
        headers["X-API-Key"] = key
    with pytest.raises(MCPAuthenticationError) as error:
        extract_mcp_context(headers)
    assert error.value.status_code == 403


@pytest.mark.parametrize("username", [None, "", "  "])
def test_trusted_headers_require_nonempty_username(monkeypatch, username):
    monkeypatch.setattr(settings, "mcp_openwebui_trust_headers", True)
    headers = _unsigned_headers()
    if username is None:
        headers.pop("X-OpenWebUI-User-Name")
    else:
        headers["X-OpenWebUI-User-Name"] = username
    with pytest.raises(MCPAuthenticationError, match="username"):
        extract_mcp_context(headers)


def test_trusted_headers_support_configured_names_and_empty_groups(monkeypatch):
    monkeypatch.setattr(settings, "mcp_openwebui_trust_headers", True)
    monkeypatch.setattr(settings, "mcp_openwebui_username_header", "X-Team-User")
    monkeypatch.setattr(settings, "mcp_openwebui_groups_header", "X-Team-Groups")
    headers = {"X-API-Key": "test-key-1", "x-team-user": "alice@example.test"}
    assert extract_mcp_context(headers).groups == []
    headers["x-team-groups"] = "finance"
    ctx = extract_mcp_context(headers)
    assert ctx.username == "alice@example.test"
    assert ctx.groups == ["finance"]


def test_trusted_headers_all_requires_separate_opt_in(monkeypatch):
    monkeypatch.setattr(settings, "mcp_openwebui_trust_headers", True)
    monkeypatch.setattr(settings, "mcp_openwebui_allow_all_group", True)
    assert extract_mcp_context(_unsigned_headers()).groups == ["finance", "ALL", "engineering"]


@pytest.mark.parametrize("header,value", [
    ("X-OpenWebUI-User-Jwt", "invalid"),
    ("X-OpenWebUI-User-Jwt", ""),
    ("Authorization", "Bearer invalid"),
    ("Authorization", ""),
    ("Authorization", "Basic not-supported"),
])
def test_invalid_credentials_cannot_fall_back_to_trusted_headers(monkeypatch, header, value):
    monkeypatch.setattr(settings, "mcp_openwebui_trust_headers", True)
    monkeypatch.setattr(settings, "mcp_openwebui_jwt_secret", "openwebui-test-secret-at-least-32-bytes")
    headers = _unsigned_headers()
    headers[header] = value
    with pytest.raises(MCPAuthenticationError):
        extract_mcp_context(headers)


def test_bearer_identity_retains_its_signed_groups_in_trusted_header_mode(monkeypatch):
    monkeypatch.setattr(settings, "mcp_openwebui_trust_headers", True)
    headers = _unsigned_headers()
    headers["Authorization"] = "Bearer " + create_token("alice", ["team"])
    ctx = extract_mcp_context(headers)
    assert ctx.username == "alice"
    assert ctx.groups == ["team"]
    assert ctx.identity_source == "sauron-jwt"


@pytest.mark.parametrize("header", ["X-OpenWebUI-User-Jwt", "Authorization"])
def test_expired_identity_cannot_fall_back_to_trusted_headers(monkeypatch, header):
    monkeypatch.setattr(settings, "mcp_openwebui_trust_headers", True)
    secret = "openwebui-test-secret-at-least-32-bytes"
    monkeypatch.setattr(settings, "mcp_openwebui_jwt_secret", secret)
    headers = _unsigned_headers()
    if header == "Authorization":
        headers[header] = "Bearer " + create_token("alice", ["team"], expiration_minutes=-5)
    else:
        then = datetime.now(timezone.utc) - timedelta(minutes=10)
        headers[header] = jwt.encode({
            "sub": "alice", "iss": "open-webui", "iat": then,
            "exp": then + timedelta(minutes=5),
        }, secret, algorithm="HS256")
    with pytest.raises(MCPAuthenticationError, match="expired"):
        extract_mcp_context(headers)
