#!/bin/bash
set -e

# HF models must be baked at image build (scripts/prefetch_hf_models.py).
# Force offline so runtime never hits huggingface.co when the bake marker exists.
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/.cache/huggingface/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/.cache/huggingface/hub}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/root/.cache/torch/sentence_transformers}"
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

# Seed categories on first startup (if DB is empty)
if [ ! -f /app/data/.seeded ]; then
    echo "First startup — seeding categories..."
    python scripts/seed_categories.py
    touch /app/data/.seeded
    echo "Seeding complete."
fi

# Run the main command
exec "$@"
