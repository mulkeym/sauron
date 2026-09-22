from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from src.mcp import server as server_module, activity_wrap
from src.mcp.answer_contract import ANSWER_TOOL, LEGACY_ANSWER_TOOL
from src.mcp.auth import MCPContext
from src.mcp.agent_registry import AgentRegistry
from src.db.schema_registry import SchemaRegistry
from src.figures import service
from src.audit import activity


@pytest.fixture
def answer_server(monkeypatch):
    context = MCPContext(username="reader", groups=["network"], api_key="test-key-1")
    monkeypatch.setattr(server_module, "current_mcp_context", lambda: context)
    monkeypatch.setattr(activity_wrap, "current_mcp_context", lambda: context)
    monkeypatch.setattr(activity, "record_query_activity", AsyncMock())
    answer = AsyncMock(return_value={"answer": "Grounded answer", "images": []})
    monkeypatch.setattr(server_module, "ask", answer)
    # Exercise tool dispatch/validation without needing an image or search store.
    monkeypatch.setattr(service, "mcp_result", AsyncMock(side_effect=lambda payload, *_: payload))
    server = server_module.create_mcp_server(MagicMock(), SchemaRegistry(), MagicMock(), AgentRegistry())
    return server, answer


@pytest.mark.asyncio
async def test_catalog_advertises_only_unambiguous_answer_tool_with_required_question(answer_server):
    server, _ = answer_server
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert ANSWER_TOOL in tools and LEGACY_ANSWER_TOOL not in tools
    schema = tools[ANSWER_TOOL].parameters
    assert schema["required"] == ["question"]
    assert schema["properties"]["question"]["type"] == "string"
    assert schema["properties"]["question"]["minLength"] == 1
    assert "original request" in schema["properties"]["question"]["description"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [ANSWER_TOOL, LEGACY_ANSWER_TOOL])
async def test_both_names_preserve_request_identity_context_and_answer(answer_server, name):
    server, answer = answer_server
    question = "How do I generate a support file for Cisco SD-WAN?"
    result = await server.call_tool(name, {"question": question, "depth": "brief", "context": "Cisco Catalyst"})
    assert result.structured_content["answer"] == "Grounded answer"
    assert answer.await_args.kwargs["question"] == question
    assert answer.await_args.kwargs["user_groups"] == ["network"]
    assert answer.await_args.kwargs["depth"] == "brief"
    assert answer.await_args.kwargs["context"] == "Cisco Catalyst"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [ANSWER_TOOL, LEGACY_ANSWER_TOOL])
@pytest.mark.parametrize("arguments", [
    {}, {"questions": [{"question": "Which aspect interests you?", "options": []}], "timeout_ms": 30000},
    {"question": ""}, {"question": "   "}, {"question": None}, {"question": ["not a string"]},
])
async def test_bad_calls_return_repair_instructions_without_searching_or_inventing_a_question(answer_server, name, arguments):
    server, answer = answer_server
    with pytest.raises(ToolError) as error:
        await server.call_tool(name, arguments)
    assert "Retry tool_answer_from_documents" in str(error.value)
    assert "user's original request" in str(error.value)
    assert "No search ran" in str(error.value)
    answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_unrelated_tool_errors_are_not_rewritten(answer_server):
    server, answer = answer_server
    async with Client(server) as client:
        result = await client.call_tool("tool_query_database", {}, raise_on_error=False)
    assert result.is_error
    assert "Missing required argument" in result.content[0].text
    assert "Retry tool_answer_from_documents" not in result.content[0].text
    answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_client_discovers_new_name_and_receives_actionable_errors(answer_server):
    server, answer = answer_server
    async with Client(server) as client:
        names = {tool.name for tool in await client.list_tools()}
        assert ANSWER_TOOL in names and LEGACY_ANSWER_TOOL not in names
        bad = await client.call_tool(LEGACY_ANSWER_TOOL, {}, raise_on_error=False)
        assert bad.is_error
        assert "Retry tool_answer_from_documents" in bad.content[0].text
        answer.assert_not_awaited()
        good = await client.call_tool(ANSWER_TOOL, {"question": "Explain government SD-WAN"})
        assert not good.is_error
        assert good.structured_content["answer"] == "Grounded answer"
