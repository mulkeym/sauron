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


def _openwebui_groups(headers: dict[str, str]) -> list[str]:
    groups = _parse_groups(headers.get(settings.mcp_openwebui_groups_header.lower(), ""))
    if not settings.mcp_openwebui_allow_all_group:
        groups = [group for group in groups if group != "ALL"]
    return groups


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

    Supported user identities:
    * Sauron's own Bearer JWT, used by direct Sauron clients.
    * OpenWebUI's signed X-OpenWebUI-User-Jwt forwarding token. OpenWebUI group
      names arrive in a separately configured templated header because its
      forwarded JWT intentionally does not contain group claims.
    * Explicitly enabled trusted OpenWebUI username/group headers, for clients
      that have not configured signed identity forwarding.

    A valid Sauron application API key is required in every case. This means
    forwarded OpenWebUI group headers are trusted only from a client that also
    possesses the dedicated application credential.
    """
    headers = _normalise_headers(headers)
    api_key = headers.get("x-api-key", "")
    if not api_key or not validate_api_key(api_key):
        raise MCPAuthenticationError("Invalid or missing API key", status_code=403)

    openwebui_token = headers.get("x-openwebui-user-jwt", "")
    if "x-openwebui-user-jwt" in headers:
        payload = _decode_openwebui_identity(openwebui_token)
        username = payload.get("email") or payload.get("name") or payload["sub"]
        return MCPContext(
            username=str(username),
            groups=_openwebui_groups(headers),
            api_key=api_key,
            agent_id=str(payload["sub"]),
            identity_source="openwebui-jwt",
        )

    auth_header = headers.get("authorization", "")
    # A supplied credential must pass validation. Never fall back to unsigned
    # headers after a missing/expired/invalid signed token in either header.
    if "authorization" not in headers and settings.mcp_openwebui_trust_headers:
        username = headers.get(settings.mcp_openwebui_username_header.lower(), "").strip()
        # Keep existing OpenWebUI connector headers working after the upstream
        # default changes to X-Sauron-Username. An explicitly configured custom
        # header is authoritative and never falls back to a different name.
        if not username and settings.mcp_openwebui_username_header.lower() == "x-sauron-username":
            username = headers.get("x-openwebui-user-name", "").strip()
        if not username:
            raise MCPAuthenticationError("Missing OpenWebUI username header")
        return MCPContext(
            username=username,
            groups=_openwebui_groups(headers),
            api_key=api_key,
            agent_id=headers.get("x-openwebui-user-id", "").strip() or username,
            identity_source="openwebui-headers",
        )
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
