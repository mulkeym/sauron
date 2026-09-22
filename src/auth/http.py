"""Deny unauthenticated requests before routing, including mounted apps."""
from urllib.parse import urlsplit

from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from src.auth.api_key import validate_api_key


def _same_origin(source: str, target: str) -> bool:
    def origin(url: str):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        return parsed.scheme, parsed.hostname, port

    try:
        source_origin = origin(source)
        return source_origin is not None and source_origin == origin(target)
    except ValueError:
        return False


class EndpointAuthenticationMiddleware:
    """Admin sessions are confined to /admin; service endpoints require keys.

    Sign-in and read-only admin assets are public; one exact PNG link route
    delegates authorization to its expiring-signature verifier. CORS
    preflight responses are handled by the outer CORSMiddleware, which never
    dispatches them to application endpoints. Pure ASGI preserves streaming
    and MCP/LLM ContextVars.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        connection = HTTPConnection(scope)
        path = scope["path"]
        root = scope.get("root_path", "").rstrip("/")
        if root and (path == root or path.startswith(root + "/")):
            path = path[len(root):]
        method = scope.get("method", "")
        from src.figures.links import LINK_PATH, TOKEN_ROUTE_PATTERN
        import re
        if (scope["type"] == "http" and method in {"GET", "HEAD"}
                and re.fullmatch(re.escape(LINK_PATH) + TOKEN_ROUTE_PATTERN, path)):
            await self.app(scope, receive, send)
            return
        is_admin = path == "/admin" or path.startswith("/admin/")
        login_url = root + "/admin/login"

        if is_admin:
            # Cookies never authorize cross-origin API reads or writes, even
            # from origins allowed by the service API's CORS policy. Ordinary
            # top-level GET navigation may have an external Referer.
            source = connection.headers.get("origin")
            if source is None and method not in {"GET", "HEAD", "OPTIONS"}:
                source = connection.headers.get("referer")
            if source is not None and not _same_origin(source, str(connection.url)):
                await self._reject(scope, receive, send, 403, "Cross-origin admin request denied")
                return

            public_login = path == "/admin/login" and method in {"GET", "HEAD", "POST"}
            public_asset = path.startswith("/admin/static/") and method in {"GET", "HEAD"}
            if not (public_login or public_asset):
                # Resolve dynamically so the routes and this guard use the
                # same live session registry (including logout/revocation).
                from src.admin.routes import _is_authenticated

                if not _is_authenticated(connection):
                    is_admin_api = path == "/admin/api" or path.startswith("/admin/api/")
                    if method in {"GET", "HEAD"} and not is_admin_api:
                        await RedirectResponse(login_url, status_code=302)(scope, receive, send)
                    else:
                        await self._reject(scope, receive, send, 401, "Admin login required")
                    return
        else:
            key = connection.headers.get("x-api-key", "")
            # OpenAI SDKs send their application key as Bearer. JWTs remain
            # user identity only: they never satisfy this application gate.
            if not key and (path == "/v1" or path.startswith("/v1/")):
                scheme, _, credential = connection.headers.get("authorization", "").partition(" ")
                if scheme.lower() == "bearer":
                    key = credential
            if not validate_api_key(key):
                await self._reject(scope, receive, send, 403, "Invalid or missing API key")
                return

        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, status: int, detail: str) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send)
