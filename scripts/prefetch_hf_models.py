#!/usr/bin/env python3
"""Bake all Hugging Face / local ML weights into the image at Docker build time.

Runtime must not need network access to HF for:
  - local embeddings (sentence-transformers)
  - CrossEncoder reranking (app setting + LanceDB default)
  - unstructured hi_res PDF layout + table structure models

Run BEFORE HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are forced on for production
images. On success writes /app/.pdf_models_ready (name kept for entrypoint).

Corporate MITM:
  SAURON_PREFETCH_INSECURE_SSL=1  — disable TLS verify for this step only
  HF_HUB_DISABLE_XET=1 (default)  — avoid xet-bridge CDN
  SAURON_PREFETCH_ALLOW_FAIL=0    — default: fail the build if bake fails
  SKIP_PDF_MODEL_PREFETCH=1       — skip entirely (NOT recommended)
"""
from __future__ import annotations

import os
import ssl
import sys
import time
import traceback
from pathlib import Path

# Markers consumed by scripts/entrypoint.sh
_READY_MARKER = Path("/app/.pdf_models_ready")
_FAILED_MARKER = Path("/app/.pdf_models_prefetch_failed")

# Full HF repos to snapshot into the hub cache (must include nomic remote-code deps).
_SNAPSHOT_REPOS: list[str] = [
    # Local embeddings — nomic pulls trust_remote_code from nomic-bert-2048
    "nomic-ai/nomic-embed-text-v1",
    "nomic-ai/nomic-bert-2048",
    # Rerankers
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-TinyBERT-L-6",
    # PDF hi_res layout + tables
    "unstructuredio/yolo_x_layout",
    "microsoft/table-transformer-structure-recognition",
    "microsoft/table-transformer-structure-recognition-v1.1-all",
]

# sentence-transformers style ids loaded via library (populates ST + hub caches)
_ST_MODELS: list[str] = [
    # config defaults — also overridable via env at build
    os.environ.get("EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1"),
    os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    # LanceDB CrossEncoderReranker default (used if model_name not passed)
    "cross-encoder/ms-marco-TinyBERT-L-6",
]


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in ("1", "true", "yes", "on")


def _mark_ready() -> None:
    _FAILED_MARKER.unlink(missing_ok=True)
    _READY_MARKER.parent.mkdir(parents=True, exist_ok=True)
    _READY_MARKER.write_text("ok\n", encoding="utf-8")
    print(f"prefetch_hf_models: wrote {_READY_MARKER}", flush=True)


def _mark_failed(reason: str) -> None:
    _READY_MARKER.unlink(missing_ok=True)
    _FAILED_MARKER.parent.mkdir(parents=True, exist_ok=True)
    _FAILED_MARKER.write_text(reason.rstrip() + "\n", encoding="utf-8")
    print(f"prefetch_hf_models: wrote {_FAILED_MARKER}", flush=True)


def _force_disable_xet() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    print("prefetch_hf_models: HF_HUB_DISABLE_XET=1", flush=True)


def _disable_ssl_verify_everywhere() -> None:
    print(
        "prefetch_hf_models: SAURON_PREFETCH_INSECURE_SSL ON — TLS verify disabled (build only)",
        flush=True,
    )
    for key in (
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "PIP_CERT",
        "AWS_CA_BUNDLE",
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
    except Exception:
        pass

    try:
        import requests

        _orig = requests.Session.request

        def _patched(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["verify"] = False
            return _orig(self, method, url, **kwargs)

        requests.Session.request = _patched  # type: ignore[method-assign]
    except Exception:
        pass

    try:
        import httpx

        _ci, _ai = httpx.Client.__init__, httpx.AsyncClient.__init__

        def _cinit(self, *a, **kw):  # type: ignore[no-untyped-def]
            kw["verify"] = False
            return _ci(self, *a, **kw)

        def _ainit(self, *a, **kw):  # type: ignore[no-untyped-def]
            kw["verify"] = False
            return _ai(self, *a, **kw)

        httpx.Client.__init__ = _cinit  # type: ignore[method-assign]
        httpx.AsyncClient.__init__ = _ainit  # type: ignore[method-assign]
        print("prefetch_hf_models: httpx verify=False", flush=True)
    except Exception as e:
        print(f"prefetch_hf_models: httpx patch skipped: {e}", flush=True)

    try:
        from huggingface_hub import configure_http_backend
        import httpx as _httpx

        configure_http_backend(
            backend_factory=lambda: _httpx.Client(
                verify=False, follow_redirects=True, timeout=300.0
            )
        )
        print("prefetch_hf_models: huggingface_hub backend verify=False", flush=True)
    except Exception as e:
        print(f"prefetch_hf_models: hub backend patch skipped: {e}", flush=True)


def _retry(label: str, fn, attempts: int = 8, base_delay: float = 2.0):
    last: BaseException | None = None
    for i in range(1, attempts + 1):
        try:
            print(f"prefetch_hf_models: {label} (attempt {i}/{attempts})", flush=True)
            return fn()
        except BaseException as e:  # noqa: BLE001
            last = e
            print(f"prefetch_hf_models: {label} failed: {e}", flush=True)
            if i >= attempts:
                break
            delay = min(base_delay * (2 ** (i - 1)), 90.0)
            print(f"prefetch_hf_models: retrying in {delay:.0f}s...", flush=True)
            time.sleep(delay)
    assert last is not None
    raise last


def _download_snapshots(attempts: int) -> None:
    from huggingface_hub import snapshot_download

    for repo in _SNAPSHOT_REPOS:
        def _one(r=repo) -> str:
            path = snapshot_download(repo_id=r)
            print(f"prefetch_hf_models: snapshot ok {r} -> {path}", flush=True)
            return path

        _retry(f"snapshot_download {repo}", _one, attempts=attempts)


def _load_sentence_transformers(attempts: int, *, local_only: bool = False) -> None:
    from sentence_transformers import SentenceTransformer, CrossEncoder

    seen: set[str] = set()
    for mid in _ST_MODELS:
        mid = (mid or "").strip()
        if not mid or mid in seen:
            continue
        seen.add(mid)

        def _load(model_id=mid) -> None:
            # Cross-encoders vs bi-encoders
            if "cross-encoder" in model_id.lower():
                m = CrossEncoder(model_id, local_files_only=local_only)
                _ = m.predict([("query", "document")])
                print(
                    f"prefetch_hf_models: CrossEncoder loaded {model_id} "
                    f"(local_only={local_only})",
                    flush=True,
                )
            else:
                m = SentenceTransformer(
                    model_id,
                    device="cpu",
                    trust_remote_code=True,
                    local_files_only=local_only,
                )
                _ = m.encode(["warmup"], show_progress_bar=False)
                dim = (
                    m.get_embedding_dimension()
                    if hasattr(m, "get_embedding_dimension")
                    else m.get_sentence_embedding_dimension()
                )
                print(
                    f"prefetch_hf_models: SentenceTransformer loaded {model_id} "
                    f"dim={dim} (local_only={local_only})",
                    flush=True,
                )

        _retry(f"load {mid}", _load, attempts=attempts)


def _verify_offline_reload() -> None:
    """Prove the hub/ST caches work with no network (catches missing nomic-bert-2048)."""
    print("prefetch_hf_models: verifying offline reload...", flush=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    # Fresh import path not required; libraries honor env on next from_pretrained.
    _load_sentence_transformers(attempts=2, local_only=True)
    print("prefetch_hf_models: offline reload OK", flush=True)


def _load_table_transformers(attempts: int) -> None:
    """Warm transformers weights used by unstructured table structure."""
    try:
        from transformers import DetrImageProcessor, TableTransformerForObjectDetection
    except Exception as e:
        print(f"prefetch_hf_models: transformers table models skipped: {e}", flush=True)
        return

    for repo in (
        "microsoft/table-transformer-structure-recognition",
        "microsoft/table-transformer-structure-recognition-v1.1-all",
    ):
        def _one(r=repo) -> None:
            _ = DetrImageProcessor.from_pretrained(r)
            _ = TableTransformerForObjectDetection.from_pretrained(r)
            print(f"prefetch_hf_models: table-transformer loaded {r}", flush=True)

        try:
            _retry(f"table-transformer {repo}", _one, attempts=max(3, attempts // 2))
        except Exception as e:
            print(f"prefetch_hf_models: table model {repo} warning: {e}", flush=True)


def _warmup_partition_pdf(attempts: int) -> None:
    smoke = "tests/fixtures/pdf/tiny_smoke.pdf"
    if not os.path.isfile(smoke):
        raise FileNotFoundError(f"missing smoke PDF {smoke}")

    from unstructured.partition.pdf import partition_pdf

    def _run():
        return partition_pdf(
            filename=smoke,
            strategy="hi_res",
            infer_table_structure=True,
        )

    _retry("partition_pdf hi_res", _run, attempts=attempts, base_delay=3.0)
    print("prefetch_hf_models: partition_pdf hi_res OK", flush=True)


def _warmup_lancedb_reranker(attempts: int) -> None:
    """LanceDB CrossEncoderReranker default model (TinyBERT)."""
    def _run():
        from lancedb.rerankers import CrossEncoderReranker

        # Instantiate so weights download; do not need a table.
        r = CrossEncoderReranker(column="text")
        print(f"prefetch_hf_models: CrossEncoderReranker model={r.model_name}", flush=True)

    _retry("CrossEncoderReranker", _run, attempts=max(3, attempts // 2))


def main() -> int:
    # Default: hard-fail so images are offline-ready. Opt out only when debugging.
    allow_fail = _truthy(os.environ.get("SAURON_PREFETCH_ALLOW_FAIL", "0"))

    if _truthy(os.environ.get("SKIP_PDF_MODEL_PREFETCH")) or _truthy(
        os.environ.get("SKIP_HF_MODEL_PREFETCH")
    ):
        print(
            "prefetch_hf_models: SKIP_* set — NOT baking HF models "
            "(runtime will need network for embeddings/rerank/OCR)",
            flush=True,
        )
        _mark_failed("skipped by SKIP_HF_MODEL_PREFETCH / SKIP_PDF_MODEL_PREFETCH")
        return 0 if allow_fail else 1

    _force_disable_xet()

    insecure = _truthy(os.environ.get("SAURON_PREFETCH_INSECURE_SSL"))
    print(
        f"prefetch_hf_models: insecure_ssl={insecure} allow_fail={allow_fail}",
        flush=True,
    )
    if insecure:
        _disable_ssl_verify_everywhere()
    else:
        ca = "/etc/ssl/certs/ca-certificates.crt"
        if os.path.isfile(ca):
            os.environ.setdefault("SSL_CERT_FILE", ca)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
            os.environ.setdefault("CURL_CA_BUNDLE", ca)

    # Prefer a stable cache location inside the image
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface/hub")
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/root/.cache/torch/sentence_transformers")

    attempts = 3 if allow_fail else 8

    try:
        _download_snapshots(attempts=attempts)
        _load_table_transformers(attempts=attempts)
        _load_sentence_transformers(attempts=attempts, local_only=False)
        _warmup_lancedb_reranker(attempts=attempts)
        _warmup_partition_pdf(attempts=attempts)
        # Must pass with HF offline — ensures nomic remote code + weights are local.
        _verify_offline_reload()
    except Exception as e:
        print(f"prefetch_hf_models FAILED: {e}", file=sys.stderr)
        traceback.print_exc()
        _mark_failed(str(e))
        if allow_fail:
            print(
                "prefetch_hf_models: SAURON_PREFETCH_ALLOW_FAIL=1 — continuing without models",
                flush=True,
            )
            return 0
        print(
            "hint: ensure network can reach huggingface.co (not just pypi). "
            "MITM: certs/Trusted_Root_CAs.pem + SAURON_PREFETCH_INSECURE_SSL=1. "
            "Or pre-seed hf-cache/ and COPY into the image.",
            file=sys.stderr,
        )
        return 1

    _mark_ready()
    print("prefetch_hf_models: ALL models baked successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
