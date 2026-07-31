#!/usr/bin/env python3
"""Backward-compatible entrypoint — full bake lives in prefetch_hf_models.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefetch_hf_models import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
