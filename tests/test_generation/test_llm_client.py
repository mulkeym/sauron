"""Typed LLM exceptions let callers distinguish a deterministic timeout
(not worth retrying) from a transient connection error (worth retrying)."""
import pytest
import requests

from src.generation import llm_client
from src.generation.llm_client import (
    LLMError, LLMTimeoutError, LLMConnectionError,
)


def test_timeout_raises_typed_timeout_error(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise requests.Timeout()
    monkeypatch.setattr(llm_client.requests, "post", raise_timeout)
    with pytest.raises(LLMTimeoutError):
        llm_client._call_llm([{"role": "user", "content": "hi"}], "m", 0.0, 8)


def test_connection_error_raises_typed_connection_error(monkeypatch):
    def raise_conn(*args, **kwargs):
        raise requests.ConnectionError()
    monkeypatch.setattr(llm_client.requests, "post", raise_conn)
    with pytest.raises(LLMConnectionError):
        llm_client._call_llm([{"role": "user", "content": "hi"}], "m", 0.0, 8)


def test_typed_errors_are_runtimeerror_subclasses():
    # Existing `except RuntimeError` / `except Exception` handlers must still catch them.
    assert issubclass(LLMTimeoutError, LLMError)
    assert issubclass(LLMConnectionError, LLMError)
    assert issubclass(LLMError, RuntimeError)
