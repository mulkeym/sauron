import pytest
from unittest.mock import patch, MagicMock
import numpy as np


def test_embed_texts_returns_vectors():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    with patch("src.ingestion.embedder._get_model", return_value=mock_model):
        from src.ingestion.embedder import embed_texts
        vectors = embed_texts(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    mock_model.encode.assert_called_once()


def test_embed_texts_adds_query_prefix():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])
    with patch("src.ingestion.embedder._get_model", return_value=mock_model):
        from src.ingestion.embedder import embed_texts
        embed_texts(["test"], prefix="query: ")
    call_args = mock_model.encode.call_args[0][0]
    assert call_args[0] == "query: test"


def test_embed_texts_empty_input():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([]).reshape(0, 0)
    with patch("src.ingestion.embedder._get_model", return_value=mock_model):
        from src.ingestion.embedder import embed_texts
        vectors = embed_texts([])
    assert len(vectors) == 0
