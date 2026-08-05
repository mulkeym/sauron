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


def extract_mcp_context(headers: dict) -> MCPContext:
    """Resolve the application and user identity for one MCP HTTP request.

    Two user-token formats are accepted:
    * Sauron's own Bearer JWT, used by direct Sauron clients.
    * OpenWebUI's signed X-OpenWebUI-User-Jwt forwarding token. OpenWebUI group
      names arrive in a separately configured templated header because its
      forwarded JWT intentionally does not contain group claims.

    A valid Sauron application API key is required in both cases. This means
    forwarded OpenWebUI group headers are trusted only from a client that also
    possesses the dedicated application credential.
    """
    headers = _normalise_headers(headers)
    api_key = headers.get("x-api-key", "")
    if not api_key or not validate_api_key(api_key):
        raise MCPAuthenticationError("Invalid or missing API key", status_code=403)

    openwebui_token = headers.get("x-openwebui-user-jwt", "")
    if openwebui_token:
        payload = _decode_openwebui_identity(openwebui_token)
        groups_header = settings.mcp_openwebui_groups_header.lower()
        groups = _parse_groups(headers.get(groups_header, ""))
        if not settings.mcp_openwebui_allow_all_group:
            groups = [group for group in groups if group != "ALL"]
        username = payload.get("email") or payload.get("name") or payload["sub"]
        return MCPContext(
            username=str(username),
            groups=groups,
            api_key=api_key,
            agent_id=str(payload["sub"]),
            identity_source="openwebui-jwt",
        )

    auth_header = headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise MCPAuthenticationError("Missing Bearer token")
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
