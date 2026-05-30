import pytest
from unittest.mock import MagicMock, patch
from src.agent.strategies.sweep import retrieve_sweep
from src.agent.state import AgentState, QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata

def _make_chunk(text, doc_id="d1", filename="transcript.txt", speaker=None, utterance_type=None, score=0.9):
    return RetrievedChunk(text=text, score=score,
        metadata=ChunkMetadata(doc_id=doc_id, filename=filename, doc_type="transcript", chunk_index=0, start_char=0, acl_groups=["engineering"], speaker=speaker, utterance_type=utterance_type))

@pytest.mark.asyncio
async def test_sweep_combines_semantic_and_metadata_results():
    d1_chunk = _make_chunk("Mike asked about Q2 timeline", doc_id="d1", speaker="Mike", utterance_type="question")
    d2_chunk = _make_chunk("Mike: What's blocking the API?", doc_id="d2", speaker="Mike", utterance_type="question")
    mock_store = MagicMock()
    # sweep discovers relevant docs via search, then fetches each doc's chunks
    mock_store.search.return_value = [d1_chunk, d2_chunk]
    mock_store.hybrid_search.return_value = []
    _by_doc = {"d1": [d1_chunk], "d2": [d2_chunk]}
    mock_store.get_chunks_by_doc.side_effect = lambda doc_id, *a, **k: _by_doc.get(doc_id, [])
    with patch("src.agent.strategies.sweep.embed_query", return_value=[0.1] * 1024):
        state = AgentState(question="What questions did Mike ask in all meetings?", user_groups=["engineering"], query_type=QueryType.SWEEP, retrieved_chunks=[], retrieval_attempts=0)
        result = await retrieve_sweep(state, vector_store=mock_store)
    assert len(result["retrieved_chunks"]) >= 1
    assert result["retrieval_attempts"] == 1

@pytest.mark.asyncio
async def test_sweep_deduplicates():
    same_chunk = _make_chunk("Mike: Are we on track?", doc_id="d1", speaker="Mike")
    mock_store = MagicMock()
    # the same doc surfaced repeatedly in discovery must dedupe to one doc fetch
    mock_store.search.return_value = [same_chunk, same_chunk]
    mock_store.hybrid_search.return_value = []
    mock_store.get_chunks_by_doc.side_effect = lambda doc_id, *a, **k: [same_chunk] if doc_id == "d1" else []
    with patch("src.agent.strategies.sweep.embed_query", return_value=[0.1] * 1024):
        state = AgentState(question="What did Mike ask?", user_groups=["engineering"], query_type=QueryType.SWEEP, retrieved_chunks=[], retrieval_attempts=0)
        result = await retrieve_sweep(state, vector_store=mock_store)
    assert len(result["retrieved_chunks"]) == 1


@pytest.mark.asyncio
async def test_sweep_returns_feedback_boosts(monkeypatch):
    """retrieve_sweep should fetch and surface feedback_boosts in its result."""
    from src.agent.strategies import sweep as sweep_mod
    from src.retrieval.models import RetrievedChunk, ChunkMetadata

    async def fake_boosts(qv, ug):
        return {"docA": 0.5}
    monkeypatch.setattr(sweep_mod, "get_feedback_boosts", fake_boosts, raising=False)
    monkeypatch.setattr(sweep_mod, "embed_query", lambda q: [0.0, 0.1])

    class FakeVS:
        def search(self, **k):
            return [RetrievedChunk(text="t", score=0.8,
                    metadata=ChunkMetadata(doc_id="docA", filename="f", doc_type="text",
                                           chunk_index=0, start_char=0, acl_groups=["ALL"]))]
        def hybrid_search(self, **k):
            return []
        def get_chunks_by_doc(self, *a, **k):
            return []

    monkeypatch.setattr(sweep_mod, "_extract_date_filter", lambda *a, **k: [])
    state = {"question": "q", "user_groups": ["ALL"]}
    result = await sweep_mod.retrieve_sweep(state, vector_store=FakeVS())
    assert result.get("feedback_boosts") == {"docA": 0.5}
