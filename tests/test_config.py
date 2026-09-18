# tests/test_config.py
import os
import pytest


def test_settings_loads_defaults(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    monkeypatch.setenv("API_KEYS", "key-a,key-b,key-c")

    # Re-import to pick up env overrides
    from src.config import Settings
    s = Settings(_env_file=None)

    assert s.jwt_secret_key == "test-secret"
    assert s.api_key_list == ["key-a", "key-b", "key-c"]
    assert s.lancedb_table_name == "chunks"
    assert s.query_cache_mode == "off"
    assert s.jwt_algorithm == "HS256"


def test_api_key_list_handles_whitespace(monkeypatch):
    monkeypatch.setenv("API_KEYS", " key-1 , key-2 , ")
    from src.config import Settings
    s = Settings(_env_file=None)
    assert s.api_key_list == ["key-1", "key-2"]
