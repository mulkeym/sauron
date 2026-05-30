import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.agent.graph import create_agent_graph, run_agent
from src.agent.state import QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata, Citation

def _make_chunk(text="Test content", score=0.9):
    return RetrievedChunk(text=text, score=score, metadata=ChunkMetadata(doc_id="d1", filename="test.pdf", doc_type="pdf", chunk_index=0, start_char=0, acl_groups=["finance"]))

def test_create_agent_graph():
    mock_store = MagicMock()
    from src.db.schema_registry import SchemaRegistry
    graph = create_agent_graph(vector_store=mock_store, schema_registry=SchemaRegistry())
    assert graph is not None


def test_create_agent_graph_without_synthesize_ends_at_merge():
    g = create_agent_graph(vector_store=MagicMock(), schema_registry=MagicMock(),
                           metadata_store=MagicMock(), include_synthesize=False)
    nodes = set(g.get_graph().nodes.keys())
    assert "merge" in nodes
    assert "synthesize" not in nodes


def test_create_agent_graph_includes_synthesize_by_default():
    g = create_agent_graph(vector_store=MagicMock(), schema_registry=MagicMock(),
                           metadata_store=MagicMock())
    assert "synthesize" in set(g.get_graph().nodes.keys())

@pytest.mark.asyncio
async def test_run_agent_lookup():
    mock_store = MagicMock()
    mock_store.hybrid_search_reranked.return_value = [_make_chunk("Policy 4.2 says expenses over $500 need approval")]
    mock_store.expand_window.side_effect = lambda chunks, window=2: chunks
    with patch("src.agent.classifier.generate", return_value='{"query_type": "lookup", "sub_tasks": ["Find policy 4.2"]}'):
        with patch("src.agent.strategies.lookup.embed_query", return_value=[0.1] * 1024), \
             patch("src.ingestion.embedder.embed_texts", side_effect=lambda texts, kind: [[0.1] * 1024 for _ in texts]):
            with patch("src.agent.synthesizer.generate", return_value="Policy 4.2 requires approval for expenses over $500 [1]."):
                from src.db.schema_registry import SchemaRegistry
                result = await run_agent(question="What is policy 4.2?", user_groups=["finance"], vector_store=mock_store, schema_registry=SchemaRegistry())
    assert "approval" in result.answer.lower() or "500" in result.answer
    assert len(result.citations) >= 1

def test_merge_results_reranks_when_enabled(monkeypatch):
    from src.retrieval.vector_store import VectorStore
    from src.config import settings

    calls = {}

    def fake_rerank(chunks, text_query, top_n, boosts=None):
        calls["args"] = (text_query, top_n, boosts)
        return chunks

    vs = VectorStore.__new__(VectorStore)
    monkeypatch.setattr(vs, "rerank_chunks", fake_rerank)
    monkeypatch.setattr(settings, "rerank_final_enabled", True)
    monkeypatch.setattr(settings, "rerank_final_top_n", 7)

    from src.agent.graph import _rerank_merge
    state = {"question": "hello", "retrieved_chunks": [1, 2], "feedback_boosts": {"d1": 0.3}}
    out = _rerank_merge(state, vs)
    assert out == {}
    assert calls["args"] == ("hello", 7, {"d1": 0.3})


def test_merge_results_noop_when_disabled(monkeypatch):
    from src.agent.graph import _rerank_merge
    from src.retrieval.vector_store import VectorStore
    from src.config import settings

    vs = VectorStore.__new__(VectorStore)
    def boom(*a, **k):
        raise AssertionError("rerank should not run when disabled")
    monkeypatch.setattr(vs, "rerank_chunks", boom)
    monkeypatch.setattr(settings, "rerank_final_enabled", False)
    out = _rerank_merge({"question": "x", "retrieved_chunks": [1]}, vs)
    assert out == {}


@pytest.mark.asyncio
async def test_run_agent_no_results():
    mock_store = MagicMock()
    mock_store.hybrid_search_reranked.return_value = []
    mock_store.expand_window.side_effect = lambda chunks, window=2: chunks
    with patch("src.agent.classifier.generate", return_value='{"query_type": "lookup", "sub_tasks": ["search"]}'):
        with patch("src.agent.strategies.lookup.embed_query", return_value=[0.1] * 1024), \
             patch("src.ingestion.embedder.embed_texts", side_effect=lambda texts, kind: [[0.1] * 1024 for _ in texts]):
            from src.db.schema_registry import SchemaRegistry
            result = await run_agent(question="Something obscure", user_groups=["finance"], vector_store=mock_store, schema_registry=SchemaRegistry())
    assert "could not find" in result.answer.lower()


@pytest.mark.asyncio
async def test_lookup_then_structured_merges_sql(monkeypatch):
    from src.agent import graph as g
    def fake_lookup(state, vector_store=None):
        return {"retrieved_chunks": ["L"], "retrieval_attempts": 1}
    async def fake_struct(state, vector_store, schema_registry):
        return {"sql_results": [{"x": 1}], "structured_trace": {"status": "ran"},
                "retrieved_chunks": ["S"]}
    monkeypatch.setattr(g, "retrieve_lookup", fake_lookup)
    monkeypatch.setattr(g, "retrieve_structured", fake_struct)
    out = await g._lookup_then_structured({"question": "q", "user_groups": ["x"]},
                                          vector_store=None, schema_registry=None)
    assert out["sql_results"] == [{"x": 1}]
    assert out["structured_trace"] == {"status": "ran"}
    assert out["retrieved_chunks"] == ["L", "S"]


@pytest.mark.asyncio
async def test_lookup_then_structured_no_table_unchanged(monkeypatch):
    from src.agent import graph as g
    def fake_lookup(state, vector_store=None):
        return {"retrieved_chunks": ["L"], "retrieval_attempts": 1}
    async def fake_struct(state, vector_store, schema_registry):
        return {}
    monkeypatch.setattr(g, "retrieve_lookup", fake_lookup)
    monkeypatch.setattr(g, "retrieve_structured", fake_struct)
    out = await g._lookup_then_structured({"question": "q", "user_groups": ["x"]},
                                          vector_store=None, schema_registry=None)
    assert "sql_results" not in out
    assert out["retrieved_chunks"] == ["L"]


@pytest.mark.asyncio
async def test_lookup_then_structured_skips_when_sql_present(monkeypatch):
    from src.agent import graph as g
    calls = {"struct": 0}
    def fake_lookup(state, vector_store=None):
        return {"retrieved_chunks": ["L"], "sql_results": [{"y": 2}]}
    async def fake_struct(state, vector_store, schema_registry):
        calls["struct"] += 1
        return {"sql_results": [{"z": 3}]}
    monkeypatch.setattr(g, "retrieve_lookup", fake_lookup)
    monkeypatch.setattr(g, "retrieve_structured", fake_struct)
    out = await g._lookup_then_structured({"question": "q", "user_groups": ["x"]},
                                          vector_store=None, schema_registry=None)
    assert out["sql_results"] == [{"y": 2}]
    assert calls["struct"] == 0


@pytest.mark.asyncio
async def test_run_agent_temporal_does_not_nameerror():
    # TEMPORAL (and the default else) branch used to reference an undefined
    # _asyncio_lookup; they now route through _lookup_then_structured like LOOKUP.
    mock_store = MagicMock()
    mock_store.hybrid_search_reranked.return_value = [_make_chunk("Something dated 2025")]
    mock_store.expand_window.side_effect = lambda chunks, window=2: chunks
    with patch("src.agent.classifier.generate", return_value='{"query_type": "temporal", "sub_tasks": ["when"]}'):
        with patch("src.agent.strategies.lookup.embed_query", return_value=[0.1] * 1024), \
             patch("src.ingestion.embedder.embed_texts", side_effect=lambda texts, kind: [[0.1] * 1024 for _ in texts]):
            with patch("src.agent.synthesizer.generate", return_value="Here is the dated info [1]."):
                from src.db.schema_registry import SchemaRegistry
                result = await run_agent(question="what changed in 2025?", user_groups=["finance"], vector_store=mock_store, schema_registry=SchemaRegistry())
    assert result.answer  # completed without NameError
