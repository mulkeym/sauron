import pytest
from unittest.mock import patch, MagicMock
import os


def test_embed_texts_returns_vectors_via_api():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6]),
    ]
    mock_client.embeddings.create.return_value = mock_response

    with patch("src.ingestion.embedder.settings") as mock_settings:
        mock_settings.embedding_mode = "api"
        mock_settings.embedding_api_url = "http://fake:8000/v1"
        mock_settings.embedding_model_name = "test-model"
        with patch("src.ingestion.embedder.OpenAI", return_value=mock_client):
            from src.ingestion.embedder import embed_texts
            vectors = embed_texts(["hello", "world"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    mock_client.embeddings.create.assert_called_once()


def test_embed_texts_adds_prefix():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2])]
    mock_client.embeddings.create.return_value = mock_response

    with patch("src.ingestion.embedder.settings") as mock_settings:
        mock_settings.embedding_mode = "api"
        mock_settings.embedding_api_url = "http://fake:8000/v1"
        mock_settings.embedding_model_name = "test-model"
        with patch("src.ingestion.embedder.OpenAI", return_value=mock_client):
            from src.ingestion.embedder import embed_texts
            embed_texts(["test"], prefix="query: ")

    call_args = mock_client.embeddings.create.call_args[1]
    assert call_args["input"][0] == "query: test"


def test_embed_texts_empty_input():
    from src.ingestion.embedder import embed_texts
    vectors = embed_texts([])
    assert len(vectors) == 0
