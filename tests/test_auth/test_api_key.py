# tests/test_auth/test_api_key.py
import pytest
from src.auth.api_key import validate_api_key


def test_valid_api_key():
    assert validate_api_key("test-key-1") is True


def test_invalid_api_key():
    assert validate_api_key("bogus-key") is False


def test_empty_api_key():
    assert validate_api_key("") is False
