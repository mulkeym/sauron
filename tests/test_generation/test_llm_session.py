"""Switchyard session/agent headers on answer-pipeline LLM calls."""
import asyncio
import uuid

import pytest

from src.generation import llm_client


def test_resolve_prefers_switchyard_session_header():
    sid, _ = llm_client.resolve_llm_identity(headers={
        "X-Switchyard-Session-Id": "sy-1",
        "X-Session-Id": "generic-1",
        "session-id": "codex-1",
        "X-OpenWebUI-Chat-Id": "owui-1",
    })
    assert sid == "sy-1"


def test_resolve_falls_back_to_openwebui_chat_id():
    sid, _ = llm_client.resolve_llm_identity(headers={
        "X-OpenWebUI-Chat-Id": "chat-99",
    })
    assert sid == "chat-99"


def test_resolve_mints_uuid_when_no_session_header():
    sid, _ = llm_client.resolve_llm_identity(headers={})
    uuid.UUID(sid)


def test_resolve_agent_prefers_switchyard_then_openwebui_then_explicit():
    _, aid = llm_client.resolve_llm_identity(
        headers={"X-OpenWebUI-User-Id": "owui-user"},
        agent_id="api-user",
    )
    assert aid == "owui-user"
    _, aid = llm_client.resolve_llm_identity(agent_id="api-user")
    assert aid == "api-user"
    _, aid = llm_client.resolve_llm_identity(headers={})
    assert aid is None


def test_unbound_outbound_headers_are_empty():
    assert llm_client.outbound_llm_headers() == {}


def test_bound_outbound_headers_include_session_request_and_agent():
    with llm_client.llm_session(session_id="sess-a", agent_id="alice"):
        h1 = llm_client.outbound_llm_headers()
        h2 = llm_client.outbound_llm_headers()
    assert h1["x-switchyard-session-id"] == "sess-a"
    assert h1["x-switchyard-agent-id"] == "alice"
    assert h1["x-switchyard-request-id"] != h2["x-switchyard-request-id"]
    uuid.UUID(h1["x-switchyard-request-id"])


def test_call_llm_sends_session_headers_when_bound(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, headers, timeout, verify):
        captured["headers"] = headers
        return FakeResp()

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    with llm_client.llm_session(session_id="sess-b", agent_id="bob"):
        llm_client._call_llm([{"role": "user", "content": "hi"}], "m", 0.0, 8)
    assert captured["headers"]["x-switchyard-session-id"] == "sess-b"
    assert captured["headers"]["x-switchyard-agent-id"] == "bob"
    assert captured["headers"]["x-switchyard-request-id"]


def test_call_llm_omits_session_headers_when_unbound(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, headers, timeout, verify):
        captured["headers"] = headers
        return FakeResp()

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    llm_client._call_llm([{"role": "user", "content": "hi"}], "m", 0.0, 8)
    assert "x-switchyard-session-id" not in captured["headers"]
    assert "x-switchyard-request-id" not in captured["headers"]


@pytest.mark.asyncio
async def test_to_thread_generate_keeps_bound_session(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, headers, timeout, verify):
        captured["headers"] = headers
        return FakeResp()

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    with llm_client.llm_session(session_id="sess-thread", agent_id="carol"):
        await asyncio.to_thread(llm_client.generate, "sys", "usr")
    assert captured["headers"]["x-switchyard-session-id"] == "sess-thread"
    assert captured["headers"]["x-switchyard-agent-id"] == "carol"


@pytest.mark.asyncio
async def test_concurrent_sessions_do_not_leak():
    seen = {}

    async def run(name):
        with llm_client.llm_session(session_id=name):
            await asyncio.sleep(0.01)
            seen[name] = llm_client.outbound_llm_headers()["x-switchyard-session-id"]

    await asyncio.gather(run("one"), run("two"))
    assert seen == {"one": "one", "two": "two"}


@pytest.mark.asyncio
async def test_agent_query_streamed_binds_session_for_cache_judge(monkeypatch):
    from src.generation.rag_chain import agent_query_streamed
    from src.retrieval.query_cache import CacheDecision

    seen = {}

    async def fake_lookup(*args, **kwargs):
        seen.update(llm_client.outbound_llm_headers())
        return CacheDecision(accepted=True, cached={"answer": "hit", "citations": []})

    monkeypatch.setattr(
        "src.generation.rag_chain.judged_cache_lookup", fake_lookup,
    )
    out = await agent_query_streamed(
        "q", ["finance"], object(), object(),
        session_id="from-api", agent_id="mike",
    )
    assert out.answer == "hit"
    assert seen["x-switchyard-session-id"] == "from-api"
    assert seen["x-switchyard-agent-id"] == "mike"


@pytest.mark.asyncio
async def test_lightrag_llm_func_forwards_extra_headers(monkeypatch):
    from src.knowledge import graph_rag

    captured = {}

    async def fake_complete(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(graph_rag, "openai_complete_if_cache", fake_complete)
    monkeypatch.setattr(graph_rag, "repair_extraction_format", lambda x: x)
    with llm_client.llm_session(session_id="kg-sess", agent_id="dana"):
        out = await graph_rag._llm_func("prompt")
    assert out == "ok"
    headers = captured["extra_headers"]
    assert headers["x-switchyard-session-id"] == "kg-sess"
    assert headers["x-switchyard-agent-id"] == "dana"
    assert headers["x-switchyard-request-id"]


@pytest.mark.asyncio
async def test_lightrag_llm_func_unbound_sends_no_session(monkeypatch):
    from src.knowledge import graph_rag

    captured = {}

    async def fake_complete(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(graph_rag, "openai_complete_if_cache", fake_complete)
    monkeypatch.setattr(graph_rag, "repair_extraction_format", lambda x: x)
    await graph_rag._llm_func("prompt")
    assert "extra_headers" not in captured


def test_mcp_session_kwargs_fail_open_without_http_request():
    from src.mcp.auth import mcp_llm_session_kwargs
    assert mcp_llm_session_kwargs() == {}


def test_llm_session_exit_across_copied_contexts_does_not_raise():
    """Starlette iterates a sync StreamingResponse one next() per thread,
    each in a fresh copy_context(). Exiting llm_session after the last
    yield must not raise 'Token was created in a different Context'."""
    import contextvars

    def gen():
        with llm_client.llm_session(session_id="sse-sess", agent_id="alice"):
            yield "a"
            yield "b"

    g = gen()
    assert contextvars.copy_context().run(next, g) == "a"
    assert contextvars.copy_context().run(next, g) == "b"
    with pytest.raises(StopIteration):
        contextvars.copy_context().run(next, g)


def test_generate_stream_explicit_session_headers(monkeypatch):
    captured = {}
    lines = [
        'data: {"choices":[{"delta":{"content":"ok"}}]}',
        "data: [DONE]",
    ]

    class FakeResp:
        def raise_for_status(self):
            pass

        def iter_lines(self, decode_unicode=True):
            return iter(lines)

    def fake_post(url, json, headers, timeout, verify, stream=False):
        captured["headers"] = headers
        return FakeResp()

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    list(llm_client.generate_stream("sys", "usr", session_id="play-1", agent_id="mike"))
    assert captured["headers"]["x-switchyard-session-id"] == "play-1"
    assert captured["headers"]["x-switchyard-agent-id"] == "mike"
