import pytest
from unittest.mock import MagicMock, patch
from src.agent.strategies.sweep import retrieve_sweep
from src.agent.state import AgentState, QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata

def _make_chunk(text, doc_id="d1", filename="transcript.txt", speaker=None, utterance_type=None, score=0.9):
    return RetrievedChunk(text=text, score=score,
        metadata=ChunkMetadata(doc_id=doc_id, filename=filename, doc_type="transcript", chunk_index=0, start_char=0, acl_groups=["engineering"], speaker=speaker, utterance_type=utterance_type))

def test_sweep_combines_semantic_and_metadata_results():
    semantic_chunks = [_make_chunk("Mike asked about Q2 timeline", speaker="Mike", utterance_type="question")]
    metadata_chunks = [_make_chunk("Mike: What's blocking the API?", doc_id="d2", speaker="Mike", utterance_type="question")]
    mock_store = MagicMock()
    mock_store.search.side_effect = [semantic_chunks, metadata_chunks]
    with patch("src.agent.strategies.sweep.embed_query", return_value=[0.1] * 1024):
        state = AgentState(question="What questions did Mike ask in all meetings?", user_groups=["engineering"], query_type=QueryType.SWEEP, retrieved_chunks=[], retrieval_attempts=0)
        result = retrieve_sweep(state, vector_store=mock_store)
    assert len(result["retrieved_chunks"]) >= 1
    assert result["retrieval_attempts"] == 1

def test_sweep_deduplicates():
    same_chunk = _make_chunk("Mike: Are we on track?", doc_id="d1", speaker="Mike")
    mock_store = MagicMock()
    mock_store.search.side_effect = [[same_chunk], [same_chunk]]
    with patch("src.agent.strategies.sweep.embed_query", return_value=[0.1] * 1024):
        state = AgentState(question="What did Mike ask?", user_groups=["engineering"], query_type=QueryType.SWEEP, retrieved_chunks=[], retrieval_attempts=0)
        result = retrieve_sweep(state, vector_store=mock_store)
    assert len(result["retrieved_chunks"]) == 1
