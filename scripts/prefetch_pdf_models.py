"""Download the unstructured hi_res layout + table-transformer models at BUILD
time so the runtime image needs no network for scanned-PDF OCR. Run during
docker build, BEFORE HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are set.

Corporate / MITM notes:
  - Hugging Face hub uses httpx; set SAURON_PREFETCH_INSECURE_SSL=1 if needed.
  - HF XET (hf-xet / xet-bridge / xet-read-token) is disabled — it breaks behind
    many proxies. We also uninstall hf-xet in the Dockerfile.
  - CDN 503s on us.aws.cdn.hf.co are retried with backoff.
  - SKIP_PDF_MODEL_PREFETCH=1 skips this step entirely (build succeeds; first
    hi_res PDF parse at runtime will need network or a pre-seeded cache).
"""
from __future__ import annotations

import os
import ssl
import sys
import time
from pathlib import Path


# Known weights pulled by unstructured hi_res + table structure (best-effort
# pre-warm before partition_pdf). Missing files are non-fatal; partition_pdf
# remains the source of truth for "models ready".
_PREFETCH_FILES: list[tuple[str, str]] = [
    ("unstructuredio/yolo_x_layout", "yolox_l0.05.onnx"),
    ("unstructuredio/yolo_x_layout", "yolox_tiny.onnx"),
    (
        "microsoft/table-transformer-structure-recognition-v1.1-all",
        "model.safetensors",
    ),
    (
        "microsoft/table-transformer-structure-recognition-v1.1-all",
        "config.json",
    ),
]


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in ("1", "true", "yes", "on")


def _disable_ssl_verify_everywhere() -> None:
    """Nuclear option for MITM proxies during build-time model bake only."""
    print(
        "prefetch_pdf_models: SAURON_PREFETCH_INSECURE_SSL is ON — "
        "disabling TLS verification for model download (build step only)",
        flush=True,
    )

    for key in (
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "PIP_CERT",
        "AWS_CA_BUNDLE",
        "HTTPLIB2_CA_CERTS",
    ):
        os.environ.pop(key, None)

    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
    try:
        ssl.create_default_context = lambda *a, **k: ssl._create_unverified_context()  # type: ignore[assignment,misc]
    except Exception:
        pass

    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception as e:
        print(f"prefetch_pdf_models: urllib3 patch skipped: {e}", flush=True)

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

        if hasattr(httpx, "request"):
            _orig_httpx_request = httpx.request

            def _httpx_request(*args, **kwargs):  # type: ignore[no-untyped-def]
                kwargs["verify"] = False
                return _orig_httpx_request(*args, **kwargs)

            httpx.request = _httpx_request  # type: ignore[assignment]

        print("prefetch_pdf_models: httpx patched verify=False", flush=True)
    except Exception as e:
        print(f"prefetch_pdf_models: httpx patch skipped: {e}", flush=True)

    try:
        from huggingface_hub import configure_http_backend
        import httpx as _httpx

        def _backend_factory() -> _httpx.Client:
            return _httpx.Client(verify=False, follow_redirects=True, timeout=120.0)

        configure_http_backend(backend_factory=_backend_factory)
        print("prefetch_pdf_models: huggingface_hub HTTP backend verify=False", flush=True)
    except Exception as e:
        print(f"prefetch_pdf_models: huggingface_hub backend patch skipped: {e}", flush=True)


def _force_disable_xet() -> None:
    """Disable HF XET before huggingface_hub is imported (constants read at import)."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    # Belt-and-suspenders: hide the package from import if still installed
    try:
        import importlib.util

        if importlib.util.find_spec("hf_xet") is not None:
            print(
                "prefetch_pdf_models: warning: hf_xet still installed; "
                "HF_HUB_DISABLE_XET=1 should skip it",
                flush=True,
            )
    except Exception:
        pass
    print("prefetch_pdf_models: HF_HUB_DISABLE_XET=1 (classic HTTPS downloads)", flush=True)


def _retry(label: str, fn, attempts: int = 8, base_delay: float = 2.0):
    last: BaseException | None = None
    for i in range(1, attempts + 1):
        try:
            print(f"prefetch_pdf_models: {label} (attempt {i}/{attempts})", flush=True)
            return fn()
        except BaseException as e:  # noqa: BLE001 — build script; retry network flakiness
            last = e
            msg = str(e)
            retryable = any(
                s in msg.lower()
                for s in (
                    "503",
                    "502",
                    "429",
                    "timeout",
                    "timed out",
                    "connection",
                    "network",
                    "temporarily",
                    "xet",
                    "ssl",
                    "reset",
                )
            )
            print(f"prefetch_pdf_models: {label} failed: {e}", flush=True)
            if i >= attempts or not retryable:
                break
            delay = min(base_delay * (2 ** (i - 1)), 60.0)
            print(f"prefetch_pdf_models: retrying in {delay:.0f}s...", flush=True)
            time.sleep(delay)
    assert last is not None
    raise last


def _httpx_download(url: str, dest: Path, verify: bool, attempts: int = 8) -> None:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _once() -> None:
        with httpx.Client(verify=verify, follow_redirects=True, timeout=300.0) as client:
            with client.stream("GET", url) as resp:
                if resp.status_code in (502, 503, 429):
                    raise RuntimeError(f"HTTP {resp.status_code} for {url}")
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(1024 * 1024):
                        f.write(chunk)
        tmp.replace(dest)
        print(f"prefetch_pdf_models: saved {dest} ({dest.stat().st_size} bytes)", flush=True)

    try:
        _retry(f"httpx GET {url}", _once, attempts=attempts)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _prewarm_hf_files(verify: bool) -> None:
    """Best-effort: pull known weights via hf_hub_download with retries, then
    fall back to raw resolve/main URLs if the hub client still hits XET CDN 503s.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HUB_DISABLE_XET

    print(f"prefetch_pdf_models: constants.HF_HUB_DISABLE_XET={HF_HUB_DISABLE_XET!r}", flush=True)

    for repo_id, filename in _PREFETCH_FILES:
        label = f"{repo_id}/{filename}"

        def _hub() -> str:
            return hf_hub_download(repo_id=repo_id, filename=filename)

        try:
            path = _retry(f"hf_hub_download {label}", _hub, attempts=5)
            print(f"prefetch_pdf_models: hub ok {path}", flush=True)
            continue
        except Exception as e:
            print(f"prefetch_pdf_models: hub failed for {label}: {e}", flush=True)

        # Fallback: classic resolve URL (still may redirect to CDN; retries help)
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        # Also try the us-east CDN-style path is automatic via redirects.
        dest = Path("/tmp/hf_manual") / repo_id.replace("/", "__") / filename
        try:
            _httpx_download(url, dest, verify=verify, attempts=6)
            # Re-try hub download — sometimes a second pass works after CDN blip;
            # if not, place file where LazyDict local-path check can find it.
            local_dir = Path(repo_id)
            local_dir.mkdir(parents=True, exist_ok=True)
            target = local_dir / filename
            if not target.exists():
                target.write_bytes(dest.read_bytes())
                print(f"prefetch_pdf_models: staged local path {target}", flush=True)
        except Exception as e:
            print(f"prefetch_pdf_models: manual download failed for {label}: {e}", flush=True)


def main() -> int:
    if _truthy(os.environ.get("SKIP_PDF_MODEL_PREFETCH")):
        print(
            "prefetch_pdf_models: SKIP_PDF_MODEL_PREFETCH set — skipping model bake",
            flush=True,
        )
        return 0

    # MUST run before any huggingface_hub import (constants bind at import time).
    _force_disable_xet()

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
        ca = "/etc/ssl/certs/ca-certificates.crt"
        if os.path.isfile(ca):
            os.environ.setdefault("SSL_CERT_FILE", ca)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
            os.environ.setdefault("CURL_CA_BUNDLE", ca)
            print(f"prefetch_pdf_models: using CA bundle {ca}", flush=True)

    verify = not insecure
    _prewarm_hf_files(verify=verify)

    from unstructured.partition.pdf import partition_pdf

    smoke = "tests/fixtures/pdf/tiny_smoke.pdf"
    if not os.path.isfile(smoke):
        print(f"prefetch_pdf_models: missing {smoke}", file=sys.stderr)
        return 1

    try:
        def _run():
            return partition_pdf(
                filename=smoke,
                strategy="hi_res",
                infer_table_structure=True,
            )

        _retry("partition_pdf hi_res", _run, attempts=6, base_delay=3.0)
    except Exception as e:
        print(f"prefetch failed: {e}", file=sys.stderr)
        print(
            "hint: CDN 503 / xet-bridge often means Hugging Face storage is "
            "unreachable from this network. Rebuild with:\n"
            "  --build-arg SAURON_PREFETCH_INSECURE_SSL=1\n"
            "or skip baking models (app still runs; hi_res needs network later):\n"
            "  --build-arg SKIP_PDF_MODEL_PREFETCH=1",
            file=sys.stderr,
        )
        return 1
    print("pdf models prefetched", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
