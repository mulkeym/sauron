#!/bin/bash
set -e

# If PDF OCR models were baked at image build time, stay offline for HF/transformers.
# If prefetch failed/skipped (common behind MITM / HF CDN 503), leave online so
# hi_res can still try at runtime when the network allows it.
if [ -f /app/.pdf_models_ready ]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    echo "PDF models ready — HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
elif [ -f /app/.pdf_models_prefetch_failed ]; then
    echo "PDF models not baked at build time (see /app/.pdf_models_prefetch_failed)."
    echo "hi_res OCR may download from Hugging Face at runtime if reachable."
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
