import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.models import RetrievedChunk, ChunkMetadata
from src.generation.rag_chain import rag_query, RAGResponse


@pytest.fixture
def mock_chunks():
    return [
        RetrievedChunk(
            text="All expenses over $500 require manager approval.",
            score=0.95,
            metadata=ChunkMetadata(
                doc_id="doc-1",
                filename="finance_policy.pdf",
                doc_type="pdf",
                chunk_index=2,
                start_char=100,
                acl_groups=["finance"],
                page=12,
            ),
        ),
        RetrievedChunk(
            text="Receipts must be submitted within 30 days.",
            score=0.88,
            metadata=ChunkMetadata(
                doc_id="doc-1",
                filename="finance_policy.pdf",
                doc_type="pdf",
                chunk_index=3,
                start_char=200,
                acl_groups=["finance"],
            ),
        ),
    ]


def test_rag_query_returns_response_with_citations(mock_chunks):
    with patch("src.generation.rag_chain.embed_query", return_value=[0.1] * 1024):
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = mock_chunks
        with patch("src.generation.rag_chain.generate", return_value="Expenses over $500 need approval [1]."):
            result = rag_query(
                question="What is the expense policy?",
                user_groups=["finance"],
                vector_store=mock_vector_store,
            )
    assert isinstance(result, RAGResponse)
    assert len(result.citations) == 2
    assert result.citations[0].filename == "finance_policy.pdf"


def test_rag_query_no_results():
    with patch("src.generation.rag_chain.embed_query", return_value=[0.1] * 1024):
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = []
        result = rag_query(
            question="Something obscure",
            user_groups=["finance"],
            vector_store=mock_vector_store,
        )
    assert "could not find" in result.answer.lower() or "no relevant" in result.answer.lower()
    assert len(result.citations) == 0
