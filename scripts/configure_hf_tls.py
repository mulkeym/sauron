#!/usr/bin/env python3
"""Configure Hugging Face HTTP clients for enterprise CA interception."""
from __future__ import annotations

import os
import ssl
from pathlib import Path


DEFAULT_CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"


def configure_huggingface_tls(
    ca_bundle: str = DEFAULT_CA_BUNDLE,
    *,
    insecure: bool = False,
) -> str:
    """Configure both Hugging Face Hub 1.x and legacy 0.x HTTP backends.

    Returns the backend name so callers can report which API was configured.
    """
    if not insecure:
        path = Path(ca_bundle)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"CA bundle is missing or empty: {ca_bundle}")
        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ[key] = ca_bundle

    import huggingface_hub

    if hasattr(huggingface_hub, "set_client_factory"):
        import httpx

        def verify_value():
            return False if insecure else ssl.create_default_context(cafile=ca_bundle)

        if hasattr(huggingface_hub, "close_session"):
            huggingface_hub.close_session()
        huggingface_hub.set_client_factory(
            lambda: httpx.Client(
                verify=verify_value(),
                follow_redirects=True,
                timeout=httpx.Timeout(300.0),
                trust_env=True,
            )
        )
        if hasattr(huggingface_hub, "set_async_client_factory"):
            huggingface_hub.set_async_client_factory(
                lambda: httpx.AsyncClient(
                    verify=verify_value(),
                    follow_redirects=True,
                    timeout=httpx.Timeout(300.0),
                    trust_env=True,
                )
            )
        return "huggingface_hub 1.x httpx client factory"

    if hasattr(huggingface_hub, "configure_http_backend"):
        import requests

        def backend_factory():
            session = requests.Session()
            session.verify = False if insecure else ca_bundle
            return session

        huggingface_hub.configure_http_backend(backend_factory=backend_factory)
        return "huggingface_hub 0.x requests backend"

    raise RuntimeError("installed huggingface_hub exposes no supported HTTP backend API")
