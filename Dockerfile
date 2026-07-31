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
ARG PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org download.pytorch.org"
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
ARG PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org download.pytorch.org"
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
RUN set -eu; \
    TH_ARGS=""; \
    for h in ${PIP_TRUSTED_HOST}; do TH_ARGS="${TH_ARGS} --trusted-host ${h}"; done; \
    echo "pip trusted-host args:${TH_ARGS}"; \
    pip install --no-cache-dir --upgrade 'pip>=26.1.2' 'setuptools>=83.0.0' wheel \
      --cert /etc/ssl/certs/ca-certificates.crt \
      ${TH_ARGS} \
 && pip install --no-cache-dir \
      torch torchvision \
      --index-url "${TORCH_CPU_INDEX}" \
      --cert /etc/ssl/certs/ca-certificates.crt \
      ${TH_ARGS} \
 && pip install --no-cache-dir \
      -r requirements.txt \
      -c constraints-security.txt \
      --cert /etc/ssl/certs/ca-certificates.crt \
      ${TH_ARGS} \
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

# Bake offline hi_res PDF/OCR models into the image and FAIL the build if they
# are not resident (offline guarantee). Runs while network is still available;
# HF_HUB_OFFLINE is set later (in the ENV block below).
#
# MITM escape hatch (HF hub uses httpx; certifi/system CAs often still fail):
#   docker build --build-arg SAURON_PREFETCH_INSECURE_SSL=1 ...
# Log must show: "SAURON_PREFETCH_INSECURE_SSL is ON" and "httpx patched".
ARG SAURON_PREFETCH_INSECURE_SSL=0
ENV SAURON_PREFETCH_INSECURE_SSL=${SAURON_PREFETCH_INSECURE_SSL}
COPY tests/fixtures/pdf/tiny_smoke.pdf tests/fixtures/pdf/tiny_smoke.pdf
# Pass the flag on the command line as well so a stale ENV layer cannot hide it.
RUN echo "build SAURON_PREFETCH_INSECURE_SSL=${SAURON_PREFETCH_INSECURE_SSL}" \
 && SAURON_PREFETCH_INSECURE_SSL=${SAURON_PREFETCH_INSECURE_SSL} \
    python scripts/prefetch_pdf_models.py

# Create data directory
RUN mkdir -p /app/data/lancedb

# Default environment
ENV LANCEDB_PATH=/app/data/lancedb \
    LANCEDB_TABLE_NAME=chunks \
    DATABASE_URL=sqlite+aiosqlite:///./data/metadata.db \
    VLLM_REQUEST_TIMEOUT=300 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8080
VOLUME /app/data

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

ENTRYPOINT ["scripts/entrypoint.sh"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
