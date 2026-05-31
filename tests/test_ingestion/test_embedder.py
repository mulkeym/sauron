import pytest
from unittest.mock import patch, MagicMock


def _mock_resp(embeddings):
    """Build a fake OpenAI-compatible /v1/embeddings HTTP response."""
    resp = MagicMock()
    resp.json.return_value = {"data": [{"embedding": e} for e in embeddings]}
    resp.raise_for_status.return_value = None
    return resp


def test_embed_texts_returns_vectors_via_api():
    with patch("src.ingestion.embedder.settings") as mock_settings:
        mock_settings.embedding_mode = "api"
        mock_settings.embedding_api_url = "http://fake:8000/v1"
        mock_settings.embedding_model_name = "test-model"
        mock_settings.ssl_verify = True
        with patch(
            "src.ingestion.embedder.requests.post",
            return_value=_mock_resp([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        ) as post:
            from src.ingestion.embedder import embed_texts
            vectors = embed_texts(["hello", "world"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    post.assert_called_once()


def test_embed_query_adds_model_prefix_via_api():
    """nomic models prepend a 'search_query: ' prefix for query embeddings."""
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["input"] = json["input"]
        return _mock_resp([[0.1, 0.2]])

    with patch("src.ingestion.embedder.settings") as mock_settings:
        mock_settings.embedding_mode = "api"
        mock_settings.embedding_api_url = "http://fake:8000/v1"
        mock_settings.embedding_model_name = "nomic-ai/nomic-embed-text-v1"
        mock_settings.ssl_verify = True
        with patch("src.ingestion.embedder.requests.post", side_effect=fake_post):
            from src.ingestion.embedder import embed_query
            embed_query("test")

    assert captured["input"][0] == "search_query: test"


def test_embed_texts_empty_input():
    from src.ingestion.embedder import embed_texts
    assert embed_texts([]) == []


def test_local_model_loads_on_cpu():
    """Local embedding must always load on CPU — no GPU dependency."""
    import src.ingestion.embedder as emb

    emb._get_local_model.cache_clear()
    try:
        with patch("src.ingestion.embedder.settings") as mock_settings:
            mock_settings.embedding_model_name = "nomic-ai/nomic-embed-text-v1"
            with patch("sentence_transformers.SentenceTransformer") as mock_st:
                emb._get_local_model()
                mock_st.assert_called_once()
                _, kwargs = mock_st.call_args
                assert kwargs["device"] == "cpu"
    finally:
        emb._get_local_model.cache_clear()
