import pytest
from unittest.mock import AsyncMock, MagicMock
from src.mcp.tools_low import search_knowledge_graph

@pytest.mark.asyncio
async def test_search_knowledge_graph_finds_entity():
    mock_store = AsyncMock()
    entity = MagicMock()
    entity.id = 1
    entity.name = "TOEE 26"
    entity.entity_type = "project"
    mock_store.search_entities.return_value = [entity]
    mock_store.get_entity_details.return_value = {
        "entity": {"id": 1, "name": "TOEE 26", "type": "project"},
        "mentions": [{"doc_id": "doc-1", "chunk_index": 0, "context_snippet": "TOEE 26 is..."}],
        "relationships": [{"related_entity": "Counter-UxS", "entity_type": "project", "relationship_type": "requires", "direction": "outgoing", "doc_id": "doc-1", "context": ""}],
    }
    result = await search_knowledge_graph(query="TOEE 26", metadata_store=mock_store)
    assert result["entity"]["name"] == "TOEE 26"
    assert len(result["relationships"]) == 1

@pytest.mark.asyncio
async def test_search_knowledge_graph_not_found():
    mock_store = AsyncMock()
    mock_store.search_entities.return_value = []
    result = await search_knowledge_graph(query="nonexistent", metadata_store=mock_store)
    assert result["entity"] is None
