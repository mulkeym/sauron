"""Shared outbound TLS policy for the API and extraction subprocesses."""
from src.config import settings

_ssl_verify_disabled = False


def apply_ssl_verify_setting() -> None:
    """Disable TLS cert verification for outbound HTTP clients when configured.

    Idempotent: safe to call on startup and again after an admin settings save.
    Re-enabling verification after a disable requires a process restart (monkey
    patches are one-way). LLM/embedding paths also pass verify=settings.ssl_verify
    explicitly, so they pick up either direction at call time.
    """
    global _ssl_verify_disabled
    if settings.ssl_verify or _ssl_verify_disabled:
        return
    import urllib3
    import requests as _requests
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _orig_request = _requests.Session.request

    def _patched_request(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        return _orig_request(self, *args, **kwargs)

    _requests.Session.request = _patched_request
    try:
        import httpx
        _orig_httpx_client_init = httpx.Client.__init__
        _orig_httpx_async_init = httpx.AsyncClient.__init__

        def _patched_httpx_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            return _orig_httpx_client_init(self, *args, **kwargs)

        def _patched_httpx_async_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            return _orig_httpx_async_init(self, *args, **kwargs)

        httpx.Client.__init__ = _patched_httpx_init
        httpx.AsyncClient.__init__ = _patched_httpx_async_init
    except ImportError:
        pass
    _ssl_verify_disabled = True
