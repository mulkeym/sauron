# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app

# System dependencies for document parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libmagic1 poppler-utils curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /install /usr/local

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
