#!/bin/bash
set -e

# Seed categories on first startup (if DB is empty)
if [ ! -f /app/data/.seeded ]; then
    echo "First startup — seeding categories..."
    python scripts/seed_categories.py
    touch /app/data/.seeded
    echo "Seeding complete."
fi

# Run the main command
exec "$@"
