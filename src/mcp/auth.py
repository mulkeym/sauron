# src/mcp/auth.py
from __future__ import annotations

from dataclasses import dataclass

import jwt

from src.auth.api_key import validate_api_key
from src.auth.jwt import decode_token
from src.config import settings


class MCPAuthenticationError(ValueError):
    """Authentication failure with an HTTP status suitable for the MCP route."""

    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class MCPContext:
    username: str
    groups: list[str]
    api_key: str
    agent_id: str = ""
    identity_source: str = "sauron-jwt"


def _normalise_headers(headers: dict) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _parse_groups(value: str) -> list[str]:
    """Parse OpenWebUI's comma-separated group-name template safely."""
    return list(dict.fromkeys(group.strip() for group in value.split(",") if group.strip()))


def _decode_openwebui_identity(token: str) -> dict:
    secret = settings.mcp_openwebui_jwt_secret
    if not secret:
        raise MCPAuthenticationError(
            "OpenWebUI identity forwarding is not configured on Sauron"
        )
    try:
        return jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            issuer="open-webui",
            leeway=30,
            options={"require": ["sub", "iss", "iat", "exp"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise MCPAuthenticationError("OpenWebUI identity token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise MCPAuthenticationError("Invalid OpenWebUI identity token") from exc


def _forwarded_groups(headers: dict) -> list[str]:
    groups_header = settings.mcp_openwebui_groups_header.lower()
    groups = _parse_groups(headers.get(groups_header, ""))
    if not settings.mcp_openwebui_allow_all_group:
        groups = [group for group in groups if group != "ALL"]
    return groups


def extract_mcp_context(headers: dict) -> MCPContext:
    """Resolve the application and user identity for one MCP HTTP request.

    A valid Sauron application API key is always required. Then:

    1. ``Authorization: Bearer`` that verifies as a Sauron JWT (direct clients).
       An invalid Bearer is an error; it does not fall through to headers.
    2. Trusted identity headers from OpenWebUI (or another key-holding client):
       username from ``mcp_openwebui_username_header``, groups from
       ``mcp_openwebui_groups_header``.

    OpenWebUI's ``X-OpenWebUI-User-Jwt`` is ignored until an IdP is wired.
    Forwarded group headers are trusted only from a client that also possesses
    the dedicated application credential.
    """
    headers = _normalise_headers(headers)
    api_key = headers.get("x-api-key", "")
    if not api_key or not validate_api_key(api_key):
        raise MCPAuthenticationError("Invalid or missing API key", status_code=403)

    auth_header = headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ")
        try:
            user = decode_token(token)
        except ValueError as exc:
            raise MCPAuthenticationError(str(exc)) from exc
        return MCPContext(
            username=user.username,
            groups=list(user.groups),
            api_key=api_key,
            identity_source="sauron-jwt",
        )

    username_header = settings.mcp_openwebui_username_header.lower()
    username = (headers.get(username_header, "") or "").strip()
    if username:
        return MCPContext(
            username=username,
            groups=_forwarded_groups(headers),
            api_key=api_key,
            agent_id=username,
            identity_source="openwebui-headers",
        )

    raise MCPAuthenticationError("Missing user identity")


def mcp_llm_session_kwargs() -> dict:
    """Session headers + agent id from the current MCP HTTP request, if any."""
    try:
        from fastmcp.server.dependencies import get_http_request
        ctx = current_mcp_context()
        req = get_http_request()
        return {
            "session_headers": getattr(req, "headers", None),
            "agent_id": ctx.agent_id or ctx.username,
        }
    except Exception:
        return {}


def current_mcp_context() -> MCPContext:
    """Return identity stored by MCPAuthenticationMiddleware for this call."""
    from fastmcp.server.dependencies import get_http_request

    try:
        request = get_http_request()
    except RuntimeError as exc:
        raise MCPAuthenticationError(
            "Authenticated HTTP transport is required for Sauron MCP tools"
        ) from exc
    context = getattr(request.state, "mcp_context", None)
    if not isinstance(context, MCPContext):
        raise MCPAuthenticationError("MCP request identity is unavailable")
    return context
