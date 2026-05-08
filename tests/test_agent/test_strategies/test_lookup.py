import pytest
from unittest.mock import MagicMock, patch
from src.agent.strategies.lookup import retrieve_lookup
from src.agent.state import AgentState, QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata

@pytest.fixture
def mock_chunks():
    return [RetrievedChunk(text="All expenses over $500 require manager approval.", score=0.95,
        metadata=ChunkMetadata(doc_id="doc-1", filename="policy.pdf", doc_type="pdf", chunk_index=0, start_char=0, acl_groups=["finance"], page=12))]

def test_lookup_returns_chunks(mock_chunks):
    mock_store = MagicMock()
    mock_store.search.return_value = mock_chunks
    with patch("src.agent.strategies.lookup.embed_query", return_value=[0.1] * 1024):
        state = AgentState(question="What is the expense policy?", user_groups=["finance"], query_type=QueryType.LOOKUP, retrieved_chunks=[], retrieval_attempts=0)
        result = retrieve_lookup(state, vector_store=mock_store)
    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieval_attempts"] == 1

def test_lookup_empty_results():
    mock_store = MagicMock()
    mock_store.search.return_value = []
    with patch("src.agent.strategies.lookup.embed_query", return_value=[0.1] * 1024):
        state = AgentState(question="Something obscure", user_groups=["finance"], query_type=QueryType.LOOKUP, retrieved_chunks=[], retrieval_attempts=0)
        result = retrieve_lookup(state, vector_store=mock_store)
    assert result["retrieved_chunks"] == []
    assert result["retrieval_attempts"] == 1
