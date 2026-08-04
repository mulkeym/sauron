# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app

# ca-certificates needed so optional custom roots can be merged into the
# system trust store before pip hits HTTPS (public or internal).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Optional custom roots: drop certs/Trusted_Root_CAs.pem in the build context.
# Primary use case: corporate MITM / TLS inspection proxies that re-sign
# outbound HTTPS (Zscaler, Palo Alto, Netskope, Blue Coat, etc.). Without
# that proxy CA in the trust store, pip fails with CERTIFICATE_VERIFY_FAILED
# even when the network is not air-gapped.
# Directory always exists (certs/.gitkeep); the .pem itself is optional.
# Run via `sh` (not ./script) so CRLF checkouts / missing +x never produce a
# cryptic "/tmp/install_trusted_root_cas.sh: not found" from a broken shebang.
COPY certs/ /tmp/certs/
COPY scripts/install_trusted_root_cas.sh /tmp/install_trusted_root_cas.sh
RUN sed -i 's/\r$//' /tmp/install_trusted_root_cas.sh \
 && echo "certs/ contents:" && ls -la /tmp/certs/ \
 && sh /tmp/install_trusted_root_cas.sh /tmp/certs/Trusted_Root_CAs.pem

# Prefer the system bundle (includes any custom roots) over certifi alone.
# PIP_CERT is required: pip does not always honor SSL_CERT_FILE for index TLS.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Explicit forward proxy (only if inspection is NOT transparent).
# Docker also accepts the usual --build-arg HTTP_PROXY=... from the client.
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG NO_PROXY=
ARG http_proxy=
ARG https_proxy=
ARG no_proxy=
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

# Package indexes (optional overrides for air-gap mirrors).
ARG PIP_INDEX_URL=
ARG PIP_EXTRA_INDEX_URL=
# MITM proxies often break pip TLS even when a custom CA is installed (pip's
# certifi path, incomplete chain, etc.). Default trusted-host list covers the
# public indexes used by this Dockerfile; override/extend via build-arg.
# Space-separated hostnames, e.g. "pypi.org files.pythonhosted.org my-pypi.local"
ARG PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org pythonhosted.org download.pytorch.org"
# Persist pip config for all subsequent pip invocations (incl. venv).
# Note: ConfigParser forbids repeated keys — use one trusted-host with a
# multi-line indented list (pip's documented form), not multiple assignments.
RUN set -eu; \
    { \
      echo "[global]"; \
      echo "cert = /etc/ssl/certs/ca-certificates.crt"; \
      if [ -n "${PIP_INDEX_URL}" ]; then echo "index-url = ${PIP_INDEX_URL}"; fi; \
      if [ -n "${PIP_EXTRA_INDEX_URL}" ]; then echo "extra-index-url = ${PIP_EXTRA_INDEX_URL}"; fi; \
      if [ -n "${PIP_TRUSTED_HOST}" ]; then \
        echo "trusted-host ="; \
        for h in ${PIP_TRUSTED_HOST}; do echo "    ${h}"; done; \
      fi; \
    } > /etc/pip.conf; \
    mkdir -p /etc/xdg/pip && cp /etc/pip.conf /etc/xdg/pip/pip.conf; \
    echo "---- /etc/pip.conf ----"; cat /etc/pip.conf; echo "-----------------------"

# Isolated venv so CPU torch is visible to the second pip install ( --prefix
# installs are not considered "installed" by a later bare pip resolve).
RUN python -m venv /opt/venv \
 && mkdir -p /opt/venv/pip \
 && cp /etc/pip.conf /opt/venv/pip/pip.conf
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_CONFIG_FILE=/etc/pip.conf

COPY requirements.txt constraints-security.txt ./

# CPU-only PyTorch. Default PyPI Linux wheels pull multi-GB nvidia-* CUDA
# packages we never need here (local embeddings + CrossEncoder run on CPU;
# LLM inference is external). Install from the official CPU wheel index first;
# with torch already satisfied, the full requirements install will not replace
# it with a CUDA build from PyPI.
#
# Override only if you mirror torch (air-gap). MITM sites can leave the default
# once certs/Trusted_Root_CAs.pem trusts the inspection proxy.
# Re-declare ARG so this RUN layer sees the values (Docker ARG scope).
ARG PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org pythonhosted.org download.pytorch.org"
ARG PIP_INDEX_URL=
ARG PIP_EXTRA_INDEX_URL=
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
RUN set -eu; \
    TH_ARGS=""; \
    for h in ${PIP_TRUSTED_HOST}; do TH_ARGS="${TH_ARGS} --trusted-host ${h}"; done; \
    IDX_ARGS=""; \
    if [ -n "${PIP_INDEX_URL}" ]; then IDX_ARGS="${IDX_ARGS} -i ${PIP_INDEX_URL}"; fi; \
    if [ -n "${PIP_EXTRA_INDEX_URL}" ]; then IDX_ARGS="${IDX_ARGS} --extra-index-url ${PIP_EXTRA_INDEX_URL}"; fi; \
    echo "pip trusted-host args:${TH_ARGS}"; \
    echo "pip index args:${IDX_ARGS}"; \
    pip install --no-cache-dir --upgrade 'pip>=26.1.2' 'setuptools>=83.0.0' wheel \
      --cert /etc/ssl/certs/ca-certificates.crt \
      ${TH_ARGS} ${IDX_ARGS} \
 && pip install --no-cache-dir \
      torch torchvision \
      --index-url "${TORCH_CPU_INDEX}" \
      --cert /etc/ssl/certs/ca-certificates.crt \
      ${TH_ARGS} \
 && pip install --no-cache-dir \
      -r requirements.txt \
      -c constraints-security.txt \
      --cert /etc/ssl/certs/ca-certificates.crt \
      ${TH_ARGS} ${IDX_ARGS} \
 && pip uninstall -y hf-xet hf_xet 2>/dev/null || true \
 && python - <<'PY'
import pathlib
import torch

print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
site = pathlib.Path(torch.__file__).resolve().parents[1]
if site.name != "site-packages":
    site = site.parent
bad = sorted(
    p.name
    for p in site.iterdir()
    if p.name.startswith(("nvidia", "cuda_")) or p.name in ("cuda", "nvidia")
)
assert not bad, f"CUDA/NVIDIA packages leaked into image: {bad}"
# Local version tag from the CPU wheel index (e.g. 2.13.0+cpu)
assert "+cpu" in torch.__version__ or not torch.cuda.is_available(), (
    f"unexpected torch build: {torch.__version__}"
)
print("OK: CPU-only torch (no nvidia-* packages)")
PY

# Merge OS CA bundle (incl. MITM roots) into certifi so huggingface_hub /
# requests / urllib3 trust the same roots as the system store.
COPY scripts/inject_system_cas_into_certifi.py /tmp/inject_system_cas_into_certifi.py
RUN python /tmp/inject_system_cas_into_certifi.py

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app

# System dependencies for document parsing.
# libgl1 + libglib2.0-0 are required by OpenCV (cv2), which unstructured hi_res
# imports for scanned-PDF OCR layout/table detection.
# ca-certificates: default public roots + optional custom roots (below).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tesseract-ocr libmagic1 poppler-utils curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Same optional custom roots as the builder (outbound LLM/embed HTTPS, etc.).
COPY certs/ /tmp/certs/
COPY scripts/install_trusted_root_cas.sh /tmp/install_trusted_root_cas.sh
RUN sed -i 's/\r$//' /tmp/install_trusted_root_cas.sh \
 && echo "certs/ contents:" && ls -la /tmp/certs/ \
 && sh /tmp/install_trusted_root_cas.sh /tmp/certs/Trusted_Root_CAs.pem \
 && rm -rf /tmp/certs /tmp/install_trusted_root_cas.sh

# Prefer the system bundle (includes any custom roots) over certifi alone.
# Same MITM CA trust as builder so runtime LLM/embed HTTPS through the proxy works.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

# Explicit proxy for runtime layer too (prefetch + later LLM calls if set).
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG NO_PROXY=
ARG http_proxy=
ARG https_proxy=
ARG no_proxy=
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

# Copy Python virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
RUN chmod +x scripts/entrypoint.sh scripts/inject_system_cas_into_certifi.py

# Re-merge system CAs into certifi on this stage (runtime OS bundle + MITM roots).
# huggingface_hub / unstructured model downloads use certifi, not only SSL_CERT_FILE.
RUN python scripts/inject_system_cas_into_certifi.py

# Bake ALL Hugging Face / local ML weights required at runtime:
#   nomic embeddings, cross-encoder rerankers, YOLOX layout, table-transformer.
# Default: hard-fail the build if any download fails (offline-ready image).
#
#   --build-arg SAURON_PREFETCH_INSECURE_SSL=1   # MITM TLS
#   --build-arg SAURON_PREFETCH_ALLOW_FAIL=1     # do not fail build (not recommended)
#   --build-arg SKIP_HF_MODEL_PREFETCH=1         # skip bake (runtime needs HF)
# Optional: pre-seed host cache into the image (air-gap friendly):
#   mkdir -p hf-cache && # copy ~/.cache/huggingface contents here
# Optional host cache is copied when present (see COPY below).
ARG SAURON_PREFETCH_INSECURE_SSL=0
ARG SKIP_PDF_MODEL_PREFETCH=0
ARG SKIP_HF_MODEL_PREFETCH=0
ARG SAURON_PREFETCH_ALLOW_FAIL=0
ARG EMBEDDING_MODEL_NAME=nomic-ai/nomic-embed-text-v1
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# Hugging Face hub (public or Artifactory / corporate mirror). Used only during
# model bake — not left as permanent runtime env for the token.
# Example Artifactory remote:
#   HF_ENDPOINT=https://artifactory.example.com/artifactory/api/huggingfaceml/huggingface-remote
ARG HF_ENDPOINT=
ARG HF_TOKEN=
ARG HUGGING_FACE_HUB_TOKEN=
ENV SAURON_PREFETCH_INSECURE_SSL=${SAURON_PREFETCH_INSECURE_SSL} \
    SKIP_PDF_MODEL_PREFETCH=${SKIP_PDF_MODEL_PREFETCH} \
    SKIP_HF_MODEL_PREFETCH=${SKIP_HF_MODEL_PREFETCH} \
    SAURON_PREFETCH_ALLOW_FAIL=${SAURON_PREFETCH_ALLOW_FAIL} \
    EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME} \
    RERANK_MODEL=${RERANK_MODEL} \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache/torch/sentence_transformers \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken

# Optional pre-seeded HF hub cache from build context (for true air-gap builds).
# Create hf-cache/ on the host with hub/ blobs from a machine that can reach HF.
COPY hf-cache/ /root/.cache/huggingface/

# Optional pre-seeded tiktoken cache for builds that cannot reach OpenAI blob
# storage. On connected builds scripts/prefetch_hf_models.py fills this path.
COPY tiktoken-cache/ /app/.cache/tiktoken/

COPY tests/fixtures/pdf/tiny_smoke.pdf tests/fixtures/pdf/tiny_smoke.pdf
# Pass HF_ENDPOINT / token only on this RUN (bake). Prefer Artifactory remote URL
# when public huggingface.co is blocked or slow. Token is not written into final ENV.
# Do not export empty HF_ENDPOINT= — hub treats that as a blank base URL and fails.
RUN set -eu; \
    echo "build SAURON_PREFETCH_INSECURE_SSL=${SAURON_PREFETCH_INSECURE_SSL} ALLOW_FAIL=${SAURON_PREFETCH_ALLOW_FAIL} SKIP=${SKIP_HF_MODEL_PREFETCH} HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co (default)}"; \
    export SAURON_PREFETCH_INSECURE_SSL="${SAURON_PREFETCH_INSECURE_SSL}" \
      SKIP_PDF_MODEL_PREFETCH="${SKIP_PDF_MODEL_PREFETCH}" \
      SKIP_HF_MODEL_PREFETCH="${SKIP_HF_MODEL_PREFETCH}" \
      SAURON_PREFETCH_ALLOW_FAIL="${SAURON_PREFETCH_ALLOW_FAIL}" \
      EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME}" \
      RERANK_MODEL="${RERANK_MODEL}" \
      HF_HUB_DISABLE_XET=1 \
      HF_HUB_ENABLE_HF_TRANSFER=0; \
    if [ -n "${HF_ENDPOINT}" ]; then export HF_ENDPOINT="${HF_ENDPOINT}"; else unset HF_ENDPOINT || true; fi; \
    if [ -n "${HF_TOKEN}" ]; then export HF_TOKEN="${HF_TOKEN}"; else unset HF_TOKEN || true; fi; \
    if [ -n "${HUGGING_FACE_HUB_TOKEN}" ]; then export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}"; \
    elif [ -n "${HF_TOKEN:-}" ]; then export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"; \
    else unset HUGGING_FACE_HUB_TOKEN || true; fi; \
    python scripts/prefetch_hf_models.py; \
    if [ "${SAURON_PREFETCH_ALLOW_FAIL}" != "1" ] && [ "${SKIP_HF_MODEL_PREFETCH}" != "1" ] && [ "${SKIP_PDF_MODEL_PREFETCH}" != "1" ]; then \
      test -f /app/.pdf_models_ready; \
    fi

# Create data directory
RUN mkdir -p /app/data/lancedb

# HF cache paths + offline by default (models baked above). Entrypoint reinforces
# offline when /app/.pdf_models_ready exists.
ENV LANCEDB_PATH=/app/data/lancedb \
    LANCEDB_TABLE_NAME=chunks \
    DATABASE_URL=sqlite+aiosqlite:///./data/metadata.db \
    VLLM_REQUEST_TIMEOUT=300 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache/torch/sentence_transformers \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken

EXPOSE 8080
VOLUME /app/data

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

ENTRYPOINT ["scripts/entrypoint.sh"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
