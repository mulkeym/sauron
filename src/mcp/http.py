"""Native Streamable HTTP transport for Sauron's mounted MCP server."""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.types import Receive, Scope, Send

from src.mcp.auth import MCPAuthenticationError, extract_mcp_context


class MCPAuthenticationMiddleware:
    """Authenticate every MCP protocol request before FastMCP handles it."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1"): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        try:
            context = extract_mcp_context(headers)
        except MCPAuthenticationError as exc:
            await self._send_error(send, exc.status_code, str(exc))
            return

        try:
            from src.api.routes_ingest import get_metadata_store
            from src.auth.api_key import touch_api_key_usage

            await touch_api_key_usage(get_metadata_store(), context.api_key)
        except Exception:
            # Authentication has already succeeded; usage timestamps are
            # intentionally best-effort, matching the REST auth dependency.
            pass

        scope.setdefault("state", {})["mcp_context"] = context
        await self.app(scope, receive, send)

    @staticmethod
    async def _send_error(send: Send, status_code: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def create_mcp_http_app(server: Any):
    """Build the mounted, authenticated, stateless Streamable HTTP app."""
    from starlette.middleware import Middleware
    from src.config import settings

    path = settings.mcp_path.strip()
    if not path.startswith("/") or path == "/":
        raise ValueError("MCP_PATH must be an absolute, non-root URL path")
    return server.http_app(
        path=path,
        transport="streamable-http",
        stateless_http=settings.mcp_stateless_http,
        json_response=True,
        middleware=[Middleware(MCPAuthenticationMiddleware)],
    )
