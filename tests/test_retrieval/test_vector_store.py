import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.vector_store import VectorStore
from src.retrieval.models import ChunkMetadata


@pytest.fixture
def mock_qdrant():
    with patch("src.retrieval.vector_store.QdrantClient") as MockClient:
        mock = MockClient.return_value
        mock.collection_exists.return_value = False
        store = VectorStore()
        yield store, mock


def test_init_creates_collection_if_not_exists(mock_qdrant):
    store, mock = mock_qdrant
    mock.collection_exists.assert_called_once()
    mock.create_collection.assert_called_once()


def test_upsert_chunks(mock_qdrant):
    store, mock = mock_qdrant
    metadata = ChunkMetadata(
        doc_id="doc-1",
        filename="test.pdf",
        doc_type="pdf",
        chunk_index=0,
        start_char=0,
        acl_groups=["finance"],
    )
    store.upsert(texts=["hello world"], vectors=[[0.1, 0.2, 0.3]], metadatas=[metadata])
    mock.upsert.assert_called_once()


def test_search_with_acl_filter(mock_qdrant):
    store, mock = mock_qdrant
    mock.query_points.return_value = MagicMock(points=[])
    results = store.search(vector=[0.1, 0.2, 0.3], user_groups=["finance"], top_k=5)
    assert results == []
    assert mock.query_points.call_args is not None


def test_delete_by_doc_id(mock_qdrant):
    store, mock = mock_qdrant
    store.delete_by_doc_id("doc-1")
    mock.delete.assert_called_once()
