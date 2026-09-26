#!/bin/bash
set -e

# HF models must be baked at image build (scripts/prefetch_hf_models.py).
# Force offline so runtime never hits huggingface.co when the bake marker exists.
export HF_HOME="${HF_HOME:-/opt/models/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/opt/models/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/opt/models/huggingface/hub}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/opt/models/sentence_transformers}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-/app/.cache/tiktoken}"

if [ -f /app/.pdf_models_ready ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    echo "HF models baked — offline mode ON (HF_HOME=${HF_HOME})"
    # Sanity: nomic hub cache dir should exist in the image
    if ! ls -d "${HF_HOME}/hub"/models--nomic-ai--nomic-embed-text-v1 >/dev/null 2>&1; then
        echo "WARNING: nomic-embed-text-v1 not found under ${HF_HOME}/hub — embeddings may fail offline"
    fi
    if ! find "${TIKTOKEN_CACHE_DIR}" -type f ! -name README.md ! -name .gitkeep -print -quit 2>/dev/null | grep -q .; then
        echo "WARNING: tiktoken cache is empty — LightRAG initialization may try network access"
    fi
elif [ -f /app/.pdf_models_prefetch_failed ]; then
    echo "WARNING: HF models were NOT baked at build time:"
    cat /app/.pdf_models_prefetch_failed 2>/dev/null || true
    echo "Runtime may try Hugging Face downloads (often fails on corporate networks)."
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
else
    echo "WARNING: no HF model bake marker found; runtime may download from Hugging Face."
fi

# The image runs as an unprivileged user. Volumes created by older (root)
# images must be re-owned once before the app can write to them.
if ! touch /app/data/.write-test 2>/dev/null; then
    echo "ERROR: /app/data is not writable by uid $(id -u)." >&2
    echo "  Docker Compose: the data-permissions service fixes this automatically;" >&2
    echo "  otherwise run once: docker run --rm --user 0 -v <volume>:/app/data --entrypoint chown IMAGE -R $(id -u):$(id -g) /app/data" >&2
    echo "  Kubernetes: set podSecurityContext.fsGroup to $(id -g)." >&2
    exit 1
fi
rm -f /app/data/.write-test

# Seed categories on first startup (if DB is empty)
if [ ! -f /app/data/.seeded ]; then
    echo "First startup — seeding categories..."
    python scripts/seed_categories.py
    touch /app/data/.seeded
    echo "Seeding complete."
fi

# Run the main command
exec "$@"
