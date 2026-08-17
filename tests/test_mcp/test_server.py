import inspect
import threading

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.mcp.auth import MCPContext
from src.mcp.server import create_mcp_server
from src.db.schema_registry import SchemaRegistry
from src.mcp.agent_registry import AgentRegistry
from src.config import settings


def test_create_mcp_server():
    server = create_mcp_server(
        vector_store=MagicMock(),
        schema_registry=SchemaRegistry(),
        metadata_store=MagicMock(),
        agent_registry=AgentRegistry(),
    )
    assert server is not None


def test_server_has_expected_name():
    server = create_mcp_server(
        vector_store=MagicMock(),
        schema_registry=SchemaRegistry(),
        metadata_store=MagicMock(),
        agent_registry=AgentRegistry(),
    )
    assert server.name == settings.mcp_server_name


def _ctx():
    return MCPContext(username="mike", groups=["finance"], api_key="test-key-1")


@pytest.mark.asyncio
async def test_logged_mcp_wrapper_records_ask():
    from src.mcp.activity_wrap import run_logged_mcp_tool
    rec = AsyncMock()
    with patch("src.mcp.activity_wrap.current_mcp_context", return_value=_ctx()), \
         patch("src.audit.activity.record_query_activity", rec):
        result = await run_logged_mcp_tool(
            tool="ask",
            query_text="What is PTO?",
            fn=AsyncMock(return_value={"answer": "ok", "query_type": "lookup", "cached": False}),
        )
    assert result["answer"] == "ok"
    rec.assert_awaited()
    kwargs = rec.await_args.kwargs
    assert kwargs["source"] == "mcp"
    assert kwargs["tool"] == "ask"
    assert kwargs["username"] == "mike"
    assert kwargs["user_groups"] == ["finance"]
    assert kwargs["query_text"] == "What is PTO?"
    assert kwargs["strategy"] == "lookup"
    assert kwargs["status"] == "ok"


@pytest.mark.asyncio
async def test_logged_mcp_wrapper_records_error_dict():
    from src.mcp.activity_wrap import run_logged_mcp_tool
    rec = AsyncMock()
    with patch("src.mcp.activity_wrap.current_mcp_context", return_value=_ctx()), \
         patch("src.audit.activity.record_query_activity", rec):
        result = await run_logged_mcp_tool(
            tool="lookup_document",
            query_text="missing.pdf",
            fn=AsyncMock(return_value={"error": "Document 'missing.pdf' not found."}),
        )
    assert "error" in result
    kwargs = rec.await_args.kwargs
    assert kwargs["status"] == "error"
    assert "not found" in kwargs["error"]


def test_get_result_is_not_wrapped():
    from src.mcp import server as mcp_server
    source = inspect.getsource(mcp_server.create_mcp_server)
    get_idx = source.index("async def tool_get_result")
    ask_idx = source.index("run_logged_mcp_tool")
    assert "run_logged_mcp_tool" not in source[get_idx:get_idx + 800]
    assert ask_idx < get_idx  # other tools are wrapped; get_result is not


def _handler_body(source: str, name: str) -> str:
    start = source.index(f"async def {name}")
    next_at = [
        idx
        for marker in ("@mcp.tool()", "@mcp.resource")
        if (idx := source.find(marker, start + 1)) != -1
    ]
    return source[start : min(next_at)] if next_at else source[start:]


def test_sync_mcp_tools_offload_via_to_thread():
    from src.mcp import server as mcp_server
    source = inspect.getsource(mcp_server.create_mcp_server)
    for name in (
        "tool_search_documents",
        "tool_lookup_document",
        "tool_search_meetings",
        "tool_list_sources",
        "tool_summarize_documents",
    ):
        assert "asyncio.to_thread" in _handler_body(source, name), name
    for name in (
        "tool_ask",
        "tool_summarize_topic",
        "tool_compare",
        "tool_query_database",
        "tool_search_knowledge_graph",
        "tool_list_documents",
        "tool_get_result",
    ):
        assert "asyncio.to_thread" not in _handler_body(source, name), name


@pytest.mark.asyncio
async def test_search_documents_wrapper_offloads_and_records():
    expected = [{"text": "snippet", "source": "a.pdf", "doc_id": "d1"}]
    rec = AsyncMock()
    loop_ident = threading.get_ident()
    seen: dict = {}

    def fake_search(**kwargs):
        seen["ident"] = threading.get_ident()
        seen["kwargs"] = kwargs
        return expected

    server = create_mcp_server(
        vector_store=MagicMock(),
        schema_registry=SchemaRegistry(),
        metadata_store=MagicMock(),
        agent_registry=AgentRegistry(),
    )
    with patch("src.mcp.activity_wrap.current_mcp_context", return_value=_ctx()), \
         patch("src.mcp.server.current_mcp_context", return_value=_ctx()), \
         patch("src.mcp.server.search_documents", side_effect=fake_search), \
         patch("src.audit.activity.record_query_activity", rec):
        result = await server.call_tool(
            "tool_search_documents",
            {"query": "PTO policy", "doc_type": "pdf", "top_k": 5},
        )
    payload = result.structured_content
    assert payload == expected or payload.get("result") == expected
    assert seen["ident"] != loop_ident
    assert seen["kwargs"]["query"] == "PTO policy"
    assert seen["kwargs"]["user_groups"] == ["finance"]
    assert seen["kwargs"]["doc_type"] == "pdf"
    assert seen["kwargs"]["top_k"] == 5
    rec.assert_awaited()
    rec_kwargs = rec.await_args.kwargs
    assert rec_kwargs["source"] == "mcp"
    assert rec_kwargs["tool"] == "search_documents"
    assert rec_kwargs["username"] == "mike"
    assert rec_kwargs["query_text"] == "PTO policy"
    assert rec_kwargs["status"] == "ok"
