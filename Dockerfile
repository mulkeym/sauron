# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app

# Isolated venv so CPU torch is visible to the second pip install ( --prefix
# installs are not considered "installed" by a later bare pip resolve).
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt constraints-security.txt ./

# CPU-only PyTorch. Default PyPI Linux wheels pull multi-GB nvidia-* CUDA
# packages we never need here (local embeddings + CrossEncoder run on CPU;
# LLM inference is external). Install from the official CPU wheel index first;
# with torch already satisfied, the full requirements install will not replace
# it with a CUDA build from PyPI.
#
# Air-gapped: mirror the CPU wheel index and build with
#   --build-arg TORCH_CPU_INDEX=https://your-mirror/.../whl/cpu
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade 'pip>=26.1.2' 'setuptools>=83.0.0' wheel \
 && pip install --no-cache-dir \
      torch torchvision \
      --index-url "${TORCH_CPU_INDEX}" \
 && pip install --no-cache-dir \
      -r requirements.txt \
      -c constraints-security.txt \
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

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app

# System dependencies for document parsing.
# libgl1 + libglib2.0-0 are required by OpenCV (cv2), which unstructured hi_res
# imports for scanned-PDF OCR layout/table detection.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libmagic1 poppler-utils curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
RUN chmod +x scripts/entrypoint.sh

# Bake offline hi_res PDF/OCR models into the image and FAIL the build if they
# are not resident (offline guarantee). Runs while network is still available;
# HF_HUB_OFFLINE is set later (in the ENV block below).
COPY tests/fixtures/pdf/tiny_smoke.pdf tests/fixtures/pdf/tiny_smoke.pdf
RUN python scripts/prefetch_pdf_models.py

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
