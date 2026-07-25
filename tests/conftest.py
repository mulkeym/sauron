# tests/conftest.py
import os
import pytest

# Override settings for tests before any imports
os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["API_KEYS"] = "test-key-1,test-key-2"
os.environ["QDRANT_HOST"] = "localhost"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/test_metadata.db"

# data/settings.json can override Settings() after construction; pin test keys.
from src.config import settings  # noqa: E402

settings.api_keys = os.environ["API_KEYS"]
settings.jwt_secret_key = os.environ["JWT_SECRET_KEY"]

