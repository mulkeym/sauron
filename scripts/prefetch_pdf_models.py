"""Download the unstructured hi_res layout + table-transformer models at BUILD
time so the runtime image needs no network for scanned-PDF OCR. Run during
docker build, BEFORE HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are set.

TLS: Hugging Face downloads often use certifi, not the OS trust store. The
Dockerfile runs inject_system_cas_into_certifi.py first. For stubborn MITM
proxies, set SAURON_PREFETCH_INSECURE_SSL=1 (build-arg) to skip verify only
during this prefetch step.
"""
from __future__ import annotations

import os
import ssl
import sys


def _maybe_disable_ssl_verify() -> None:
    flag = os.environ.get("SAURON_PREFETCH_INSECURE_SSL", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return
    print(
        "prefetch_pdf_models: SAURON_PREFETCH_INSECURE_SSL set — "
        "disabling TLS verify for model download only",
        file=sys.stderr,
    )
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass
    try:
        import requests

        _orig = requests.Session.request

        def _patched(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs.setdefault("verify", False)
            return _orig(self, *args, **kwargs)

        requests.Session.request = _patched  # type: ignore[method-assign]
    except Exception:
        pass


def main() -> int:
    # Point common HTTP stacks at the system bundle (includes MITM roots).
    ca = "/etc/ssl/certs/ca-certificates.crt"
    if os.path.isfile(ca):
        os.environ.setdefault("SSL_CERT_FILE", ca)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
        os.environ.setdefault("CURL_CA_BUNDLE", ca)

    _maybe_disable_ssl_verify()

    # partition_pdf with hi_res pulls the layout + table models on first use;
    # invoking it once here caches them into the image layer.
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
            "hint: ensure certs/Trusted_Root_CAs.pem has your MITM inspection CA; "
            "or rebuild with --build-arg SAURON_PREFETCH_INSECURE_SSL=1",
            file=sys.stderr,
        )
        return 1
    print("pdf models prefetched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
