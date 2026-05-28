import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.agent.strategies.cross_reference import retrieve_cross_reference
from src.agent.state import AgentState, QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata
from src.db.schema_registry import SchemaRegistry, TableSchema, ColumnSchema


def _make_chunk(text, doc_type="pdf", filename="policy.pdf"):
    return RetrievedChunk(
        text=text,
        score=0.9,
        metadata=ChunkMetadata(
            doc_id="d1",
            filename=filename,
            doc_type=doc_type,
            chunk_index=0,
            start_char=0,
            acl_groups=["finance"],
        ),
    )


@pytest.fixture
def registry():
    r = SchemaRegistry()
    r.register(TableSchema(
        database="finance_db",
        table="expenses",
        columns=[ColumnSchema(name="amount", dtype="numeric", description="Amount in USD")],
        description="Expense records",
        acl_groups=["finance"],
    ))
    return r


@pytest.mark.asyncio
async def test_cross_reference_combines_doc_and_sql(registry):
    mock_store = MagicMock()
    mock_store.search.return_value = [_make_chunk("Policy 4.2: Expenses over $500 need approval")]
    with patch("src.agent.strategies.cross_reference.embed_query", return_value=[0.1] * 1024):
        with patch(
            "src.agent.strategies.cross_reference.retrieve_analytical",
            new_callable=AsyncMock,
            return_value={"sql_results": [{"amount": 750}], "retrieval_attempts": 1},
        ):
            state = AgentState(
                question="Does our spending comply with policy 4.2?",
                user_groups=["finance"],
                query_type=QueryType.CROSS_REFERENCE,
                sub_tasks=["Get spending data", "Find policy 4.2"],
                retrieved_chunks=[],
                sql_results=[],
                retrieval_attempts=0,
            )
            result = await retrieve_cross_reference(state, vector_store=mock_store, schema_registry=registry)
    assert len(result["retrieved_chunks"]) > 0
    assert len(result["sql_results"]) > 0


@pytest.mark.asyncio
async def test_cross_reference_returns_feedback_boosts(monkeypatch):
    from src.agent.strategies import cross_reference as xr
    from src.retrieval.models import RetrievedChunk, ChunkMetadata

    def fake_embed(texts, kind):
        return [[0.0, 0.1] for _ in texts]
    monkeypatch.setattr("src.ingestion.embedder.embed_texts", fake_embed)

    async def fake_boosts(qv, ug):
        return {"docB": 0.6}
    monkeypatch.setattr(xr, "get_feedback_boosts", fake_boosts, raising=False)

    def _c(doc_id, idx, score):
        return RetrievedChunk(text="t", score=score,
            metadata=ChunkMetadata(doc_id=doc_id, filename="f", doc_type="text",
                                   chunk_index=idx, start_char=0, acl_groups=["ALL"]))

    class FakeVS:
        def hybrid_search_reranked(self, **k):
            return [_c("docA", 0, 0.9), _c("docB", 1, 0.5)]
        def expand_window(self, chunks, window=2):
            return chunks

    class FakeRegistry:
        def list_for_user(self, ug):
            return []

    state = {"question": "q", "user_groups": ["ALL"], "sub_tasks": ["q"]}
    result = await xr.retrieve_cross_reference(state, vector_store=FakeVS(), schema_registry=FakeRegistry())
    assert result["feedback_boosts"] == {"docB": 0.6}
    assert result["retrieved_chunks"][0].metadata.doc_id == "docB"  # 0.5+0.6 > 0.9


@pytest.mark.asyncio
async def test_cross_reference_doc_only_when_no_db():
    mock_store = MagicMock()
    mock_store.search.return_value = [_make_chunk("Some policy content")]
    empty_registry = SchemaRegistry()
    with patch("src.agent.strategies.cross_reference.embed_query", return_value=[0.1] * 1024):
        state = AgentState(
            question="Compare policy A with policy B",
            user_groups=["finance"],
            query_type=QueryType.CROSS_REFERENCE,
            sub_tasks=["Find policy A", "Find policy B"],
            retrieved_chunks=[],
            sql_results=[],
            retrieval_attempts=0,
        )
        result = await retrieve_cross_reference(state, vector_store=mock_store, schema_registry=empty_registry)
    assert len(result["retrieved_chunks"]) > 0
    assert result["sql_results"] == []
