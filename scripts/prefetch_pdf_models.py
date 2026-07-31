"""Download the unstructured hi_res layout + table-transformer models at BUILD
time so the runtime image needs no network for scanned-PDF OCR. Run during
docker build, BEFORE HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are set.

TLS notes (corporate MITM):
  - Hugging Face hub downloads use **httpx**, not requests — system CA env
    vars and certifi merges are not always enough.
  - Set SAURON_PREFETCH_INSECURE_SSL=1 to disable TLS verify for this step only
    (patches ssl, urllib3, requests, and httpx).
"""
from __future__ import annotations

import os
import ssl
import sys


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in ("1", "true", "yes", "on")


def _disable_ssl_verify_everywhere() -> None:
    """Nuclear option for MITM proxies during build-time model bake only."""
    print(
        "prefetch_pdf_models: SAURON_PREFETCH_INSECURE_SSL is ON — "
        "disabling TLS verification for model download (build step only)",
        flush=True,
    )

    # Stop client libs from preferring a "real" CA bundle that rejects the proxy.
    for key in (
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "PIP_CERT",
        "AWS_CA_BUNDLE",
        "HTTPLIB2_CA_CERTS",
    ):
        os.environ.pop(key, None)

    # stdlib
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
    try:
        ssl.create_default_context = lambda *a, **k: ssl._create_unverified_context()  # type: ignore[assignment,misc]
    except Exception:
        pass

    # urllib3
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        try:
            from urllib3.util import ssl_ as urllib3_ssl

            urllib3_ssl.DEFAULT_CERT_REQUIREMENTS = None  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception as e:
        print(f"prefetch_pdf_models: urllib3 patch skipped: {e}", flush=True)

    # requests (some paths still use it)
    try:
        import requests

        _orig_req = requests.Session.request

        def _req_patched(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["verify"] = False
            return _orig_req(self, method, url, **kwargs)

        requests.Session.request = _req_patched  # type: ignore[method-assign]
        requests.sessions.Session.request = _req_patched  # type: ignore[method-assign]
    except Exception as e:
        print(f"prefetch_pdf_models: requests patch skipped: {e}", flush=True)

    # httpx — used by modern huggingface_hub (yolox_l0.05.onnx download path)
    try:
        import httpx

        _orig_client_init = httpx.Client.__init__
        _orig_async_init = httpx.AsyncClient.__init__

        def _client_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["verify"] = False
            return _orig_client_init(self, *args, **kwargs)

        def _async_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["verify"] = False
            return _orig_async_init(self, *args, **kwargs)

        httpx.Client.__init__ = _client_init  # type: ignore[method-assign]
        httpx.AsyncClient.__init__ = _async_init  # type: ignore[method-assign]

        # Also patch request helpers that construct ephemeral clients
        if hasattr(httpx, "request"):
            _orig_httpx_request = httpx.request

            def _httpx_request(*args, **kwargs):  # type: ignore[no-untyped-def]
                kwargs["verify"] = False
                return _orig_httpx_request(*args, **kwargs)

            httpx.request = _httpx_request  # type: ignore[assignment]

        print("prefetch_pdf_models: httpx patched verify=False", flush=True)
    except Exception as e:
        print(f"prefetch_pdf_models: httpx patch skipped: {e}", flush=True)

    # huggingface_hub backend hook (when available)
    try:
        from huggingface_hub import configure_http_backend
        import httpx as _httpx

        def _backend_factory() -> _httpx.Client:
            return _httpx.Client(verify=False, follow_redirects=True, timeout=60.0)

        configure_http_backend(backend_factory=_backend_factory)
        print("prefetch_pdf_models: huggingface_hub HTTP backend verify=False", flush=True)
    except Exception as e:
        print(f"prefetch_pdf_models: huggingface_hub backend patch skipped: {e}", flush=True)


def main() -> int:
    raw_flag = os.environ.get("SAURON_PREFETCH_INSECURE_SSL")
    print(
        f"prefetch_pdf_models: SAURON_PREFETCH_INSECURE_SSL={raw_flag!r} "
        f"(truthy={_truthy(raw_flag)})",
        flush=True,
    )

    insecure = _truthy(raw_flag)
    if insecure:
        _disable_ssl_verify_everywhere()
    else:
        # Prefer system bundle (includes MITM roots when Trusted_Root_CAs.pem present).
        ca = "/etc/ssl/certs/ca-certificates.crt"
        if os.path.isfile(ca):
            os.environ.setdefault("SSL_CERT_FILE", ca)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
            os.environ.setdefault("CURL_CA_BUNDLE", ca)
            print(f"prefetch_pdf_models: using CA bundle {ca}", flush=True)

    # Import AFTER SSL patches so clients created at import time see them.
    from unstructured.partition.pdf import partition_pdf

    try:
        partition_pdf(
            filename="tests/fixtures/pdf/tiny_smoke.pdf",
            strategy="hi_res",
            infer_table_structure=True,
        )
    except Exception as e:
        print(f"prefetch failed: {e}", file=sys.stderr)
        print(
            "hint: rebuild with --build-arg SAURON_PREFETCH_INSECURE_SSL=1 "
            "(HF downloads use httpx; must see 'httpx patched' in the log). "
            "Or ensure certs/Trusted_Root_CAs.pem is your MITM inspection CA.",
            file=sys.stderr,
        )
        return 1
    print("pdf models prefetched", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
