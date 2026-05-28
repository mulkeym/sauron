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

def test_lookup_applies_and_returns_feedback_boosts(monkeypatch):
    from src.agent.strategies import lookup as lk
    from src.retrieval.models import RetrievedChunk, ChunkMetadata

    monkeypatch.setattr(lk, "embed_query", lambda q: [0.0, 0.1])
    monkeypatch.setattr(lk, "get_feedback_boosts_sync", lambda qv, ug: {"docB": 0.4})
    # neutralize the date-filter import side path
    import src.agent.strategies.sweep as sweep_mod
    monkeypatch.setattr(sweep_mod, "_extract_date_filter", lambda *a, **k: [])

    def _c(doc_id, idx, score):
        return RetrievedChunk(text="t", score=score,
            metadata=ChunkMetadata(doc_id=doc_id, filename="f", doc_type="text",
                                   chunk_index=idx, start_char=0, acl_groups=["ALL"]))

    class FakeVS:
        def hybrid_search_reranked(self, **k):
            return [_c("docA", 0, 0.9), _c("docB", 1, 0.7)]
        def expand_window(self, chunks, window=2):
            return chunks

    state = {"question": "q", "user_groups": ["ALL"]}
    result = lk.retrieve_lookup(state, vector_store=FakeVS())
    assert result["feedback_boosts"] == {"docB": 0.4}
    # docB boosted 0.7+0.4=1.1 should now lead docA (0.9)
    assert result["retrieved_chunks"][0].metadata.doc_id == "docB"
