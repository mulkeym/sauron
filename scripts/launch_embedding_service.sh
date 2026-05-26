#!/bin/bash
set -euo pipefail

# Launch the Sauron-compatible GPU embedding service (llama.cpp).
#
# Serves nomic-embed-text-v1 (768-dim, mean-pooled, L2-normalized) over an
# OpenAI-compatible /v1/embeddings endpoint — the same model Sauron runs on
# CPU in the container. Sauron applies the search_document:/search_query:
# prefixes client-side, so the server only needs to embed raw text.
#
# Point Sauron at it via:
#   EMBEDDING_MODE=api
#   EMBEDDING_API_URL=http://host.docker.internal:${PORT}/v1
#   EMBEDDING_MODEL_NAME=nomic-ai/nomic-embed-text-v1

# --- Config (override via env) -------------------------------------------------
CONTAINER_NAME="${CONTAINER_NAME:-sauron-embed}"
GPU_DEVICE="${GPU_DEVICE:-1}"                       # 3070 Ti
PORT="${PORT:-8180}"
MODELS_DIR="${MODELS_DIR:-/home/mike/models}"
MODEL_FILE="${MODEL_FILE:-nomic-embed-text-v1.f32.gguf}"
IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:full-cuda}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/nomic-ai/nomic-embed-text-v1-GGUF/resolve/main/nomic-embed-text-v1.f32.gguf?download=true}"

MODEL_PATH="${MODELS_DIR}/${MODEL_FILE}"

# --- Download model if missing -------------------------------------------------
if [ ! -f "${MODEL_PATH}" ]; then
    echo "Model not found at ${MODEL_PATH} — downloading..."
    curl -L -f --retry 3 -o "${MODEL_PATH}" "${MODEL_URL}"
fi

# --- (Re)launch container ------------------------------------------------------
echo "Removing any existing '${CONTAINER_NAME}' container..."
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "Starting '${CONTAINER_NAME}' on GPU ${GPU_DEVICE}, port ${PORT}..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --gpus "\"device=${GPU_DEVICE}\"" \
    -p "${PORT}:8080" \
    -v "${MODELS_DIR}:/models" \
    "${IMAGE}" \
    -s -m "/models/${MODEL_FILE}" \
    --embeddings \
    --pooling mean \
    -ngl 999 \
    --host 0.0.0.0 --port 8080 \
    -c 32768 --batch-size 8192 --ubatch-size 8192 \
    --parallel 4 --cont-batching \
    `# -c is split across --parallel slots (32768/4 = 8192 tokens each).` \
    `# The GGUF declares context_length=2048, so llama.cpp caps each slot` \
    `# to 2048 unless overridden. nomic-embed-text-v1 is really an 8192` \
    `# model via RoPE/NTK scaling (the CPU sentence-transformers path uses` \
    `# it too), so override the metadata and enable YaRN to match.` \
    --override-kv nomic-bert.context_length=int:8192 \
    --rope-scaling yarn --yarn-orig-ctx 2048

# --- Wait for health -----------------------------------------------------------
echo -n "Waiting for service to become healthy"
for _ in $(seq 1 30); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo " — OK"
        echo "Embedding service ready at http://localhost:${PORT}/v1/embeddings"
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo ""
echo "ERROR: service did not become healthy in time. Recent logs:" >&2
docker logs --tail 30 "${CONTAINER_NAME}" >&2
exit 1
