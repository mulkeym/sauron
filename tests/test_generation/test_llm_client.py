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


def test_connect_timeout_raises_typed_timeout_error(monkeypatch):
    # ConnectTimeout subclasses BOTH ConnectionError and Timeout; it must be
    # classified as a (permanent) timeout, not a (transient) connection error.
    def raise_connect_timeout(*args, **kwargs):
        raise requests.ConnectTimeout()
    monkeypatch.setattr(llm_client.requests, "post", raise_connect_timeout)
    with pytest.raises(LLMTimeoutError):
        llm_client._call_llm([{"role": "user", "content": "hi"}], "m", 0.0, 8)


def test_http_error_raises_base_llm_error(monkeypatch):
    def raise_http(*args, **kwargs):
        raise requests.HTTPError()
    monkeypatch.setattr(llm_client.requests, "post", raise_http)
    with pytest.raises(LLMError) as exc_info:
        llm_client._call_llm([{"role": "user", "content": "hi"}], "m", 0.0, 8)
    assert type(exc_info.value) is LLMError  # base class, not a subclass


def test_call_llm_includes_seed(monkeypatch):
    """_call_llm sends a deterministic seed in the request payload."""
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, headers, timeout, verify):
        captured["payload"] = json
        return FakeResp()

    monkeypatch.setattr("src.generation.llm_client.requests.post", fake_post)
    from src.generation.llm_client import _call_llm
    _call_llm([{"role": "user", "content": "hi"}], model="m", temperature=0.0, max_tokens=10)
    assert captured["payload"]["seed"] == 0


def test_call_llm_thinking_adds_reasoning_toggle(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "sql_thinking_max_tokens", 4096)
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "SELECT 1"}}]}

    def fake_post(url, json, headers, timeout, verify):
        captured["payload"] = json
        return FakeResp()

    monkeypatch.setattr("src.generation.llm_client.requests.post", fake_post)
    from src.generation.llm_client import _call_llm
    out = _call_llm([{"role": "user", "content": "x"}], model="m",
                    temperature=0.0, max_tokens=2048, thinking=True)
    assert out == "SELECT 1"
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["payload"]["max_tokens"] == 4096


def test_call_llm_no_thinking_by_default(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, headers, timeout, verify):
        captured["payload"] = json
        return FakeResp()

    monkeypatch.setattr("src.generation.llm_client.requests.post", fake_post)
    from src.generation.llm_client import _call_llm
    _call_llm([{"role": "user", "content": "x"}], model="m", temperature=0.0, max_tokens=2048)
    assert "chat_template_kwargs" not in captured["payload"]
    assert captured["payload"]["max_tokens"] == 2048


def test_generate_forwards_thinking(monkeypatch):
    import src.generation.llm_client as llm

    seen = {}

    def fake_call(messages, model, temperature, max_tokens, thinking=False):
        seen["thinking"] = thinking
        return "SELECT 1"

    monkeypatch.setattr(llm, "_call_llm", fake_call)
    out = llm.generate("sys", "usr", thinking=True)
    assert out == "SELECT 1"
    assert seen["thinking"] is True
