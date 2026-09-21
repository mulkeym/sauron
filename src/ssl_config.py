"""Shared outbound TLS policy for the API and extraction subprocesses."""
import importlib

from src.config import settings

_ssl_verify_disabled = False


def _patch_http_client(client_class) -> None:
    original_init = client_class.__init__

    def configured_init(self, *args, **kwargs):
        # Read at construction time so newly created SDK clients also honor
        # re-enabling verification. Explicit per-client policies take priority.
        kwargs.setdefault("verify", settings.ssl_verify)
        return original_init(self, *args, **kwargs)

    client_class.__init__ = configured_init


def apply_ssl_verify_setting() -> None:
    """Disable TLS cert verification for outbound HTTP clients when configured.

    Idempotent: safe to call on startup and again after an admin settings save.
    Wrappers are installed once and read the current setting for new clients
    (or each requests call). Existing HTTPX clients retain their original TLS
    context; LightRAG creates a new SDK client on every attempt.
    """
    global _ssl_verify_disabled
    if settings.ssl_verify or _ssl_verify_disabled:
        return
    import urllib3
    import requests as _requests
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _orig_request = _requests.Session.request

    def _patched_request(self, *args, **kwargs):
        kwargs.setdefault("verify", settings.ssl_verify)
        return _orig_request(self, *args, **kwargs)

    _requests.Session.request = _patched_request
    # OpenAI SDK 3.x switched its default transport from httpx to httpx2.
    # Both are used in supported installations; patch the actual constructors
    # rather than assuming that the SDK still inherits from httpx.AsyncClient.
    for module_name in ("httpx", "httpx2"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        _patch_http_client(module.Client)
        _patch_http_client(module.AsyncClient)
    _ssl_verify_disabled = True
