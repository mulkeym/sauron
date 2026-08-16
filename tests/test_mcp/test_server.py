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
    import inspect
    from src.mcp import server as mcp_server
    source = inspect.getsource(mcp_server.create_mcp_server)
    get_idx = source.index("async def tool_get_result")
    ask_idx = source.index("run_logged_mcp_tool")
    assert "run_logged_mcp_tool" not in source[get_idx:get_idx + 800]
    assert ask_idx < get_idx  # other tools are wrapped; get_result is not
