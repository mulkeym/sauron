# tests/conftest.py
import os
import pytest

# Override settings for tests before any imports
os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["API_KEYS"] = "test-key-1,test-key-2"
os.environ["QDRANT_HOST"] = "localhost"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/test_metadata.db"
