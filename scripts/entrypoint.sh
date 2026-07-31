#!/bin/bash
set -e

# HF models must be baked at image build (scripts/prefetch_hf_models.py).
# When ready, force offline so runtime never hits huggingface.co.
if [ -f /app/.pdf_models_ready ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "HF models baked — HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
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
