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

# Create data directory
RUN mkdir -p /app/data/lancedb

# Default environment
ENV LANCEDB_PATH=/app/data/lancedb \
    LANCEDB_TABLE_NAME=chunks \
    DATABASE_URL=sqlite+aiosqlite:///./data/metadata.db \
    VLLM_REQUEST_TIMEOUT=300

EXPOSE 8080
VOLUME /app/data

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
