import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.retriever import retrieve
from src.retrieval.models import RetrievedChunk, ChunkMetadata


def test_retrieve_calls_embed_and_search():
    mock_store = MagicMock()
    mock_store.search.return_value = [
        RetrievedChunk(
            text="test chunk",
            score=0.9,
            metadata=ChunkMetadata(
                doc_id="d1",
                filename="a.pdf",
                doc_type="pdf",
                chunk_index=0,
                start_char=0,
                acl_groups=["finance"],
            ),
        )
    ]
    with patch("src.retrieval.retriever.embed_query", return_value=[0.1] * 1024):
        results = retrieve(query="test question", user_groups=["finance"], vector_store=mock_store, top_k=5)
    assert len(results) == 1
    assert results[0].text == "test chunk"


def test_retrieve_filters_low_score():
    mock_store = MagicMock()
    mock_store.search.return_value = [
        RetrievedChunk(
            text="good",
            score=0.9,
            metadata=ChunkMetadata(
                doc_id="d1",
                filename="a.pdf",
                doc_type="pdf",
                chunk_index=0,
                start_char=0,
                acl_groups=["finance"],
            ),
        ),
        RetrievedChunk(
            text="bad",
            score=0.2,
            metadata=ChunkMetadata(
                doc_id="d2",
                filename="b.pdf",
                doc_type="pdf",
                chunk_index=0,
                start_char=0,
                acl_groups=["finance"],
            ),
        ),
    ]
    with patch("src.retrieval.retriever.embed_query", return_value=[0.1] * 1024):
        results = retrieve(query="test", user_groups=["finance"], vector_store=mock_store, min_score=0.5)
    assert len(results) == 1
    assert results[0].text == "good"
