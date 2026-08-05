import pytest
from unittest.mock import AsyncMock, patch
from src.mcp.tools_low import search_knowledge_graph

@pytest.mark.asyncio
async def test_search_knowledge_graph_finds_entity():
    with patch(
        "src.knowledge.graph_rag.query_graph",
        new_callable=AsyncMock,
        return_value={"context": "TOEE 26 requires Counter-UxS", "mode": "local"},
    ) as query_graph:
        result = await search_knowledge_graph(
            query="TOEE 26", user_groups=["engineering"]
        )
    assert result["query"] == "TOEE 26"
    assert "Counter-UxS" in result["context"]
    query_graph.assert_awaited_once_with(
        "TOEE 26", mode="local", user_groups=["engineering"]
    )

@pytest.mark.asyncio
async def test_search_knowledge_graph_not_found():
    with patch(
        "src.knowledge.graph_rag.query_graph",
        new_callable=AsyncMock,
        return_value={"context": "", "mode": "local"},
    ):
        result = await search_knowledge_graph(
            query="nonexistent", user_groups=["finance"]
        )
    assert result["context"] == ""
