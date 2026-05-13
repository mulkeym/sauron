import json
import logging
import subprocess
from functools import lru_cache

from src.config import settings

logger = logging.getLogger(__name__)

# Prefix mappings per model family
MODEL_PREFIXES = {
    "nomic": {"passage": "search_document: ", "query": "search_query: "},
    "e5": {"passage": "passage: ", "query": "query: "},
    "default": {"passage": "", "query": ""},
}


def _get_prefixes() -> dict:
    """Get the right prefixes for the configured model."""
    model = settings.embedding_model_name.lower()
    if "nomic" in model:
        return MODEL_PREFIXES["nomic"]
    elif "e5" in model:
        return MODEL_PREFIXES["e5"]
    return MODEL_PREFIXES["default"]


def _embed_single(text: str) -> list[float]:
    """Embed a single text via API."""
    payload = {
        "model": settings.embedding_model_name,
        "input": text,
    }
    result = subprocess.run(
        ['curl', '-4', '-s', '-X', 'POST',
         f'{settings.embedding_api_url}/embeddings',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(f"Embedding request failed: {result.stderr}")
    response = json.loads(result.stdout)
    if 'error' in response:
        raise RuntimeError(f"Embedding error: {response['error']}")
    return response['data'][0]['embedding']


def _embed_via_api(texts: list[str]) -> list[list[float]]:
    """Call an OpenAI-compatible /v1/embeddings endpoint with IPv4 forcing."""
    payload = {
        "model": settings.embedding_model_name,
        "input": texts,
    }
    result = subprocess.run(
        ['curl', '-4', '-s', '-X', 'POST',
         f'{settings.embedding_api_url}/embeddings',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=120
    )
    if result.returncode == 0:
        response = json.loads(result.stdout)
        if 'data' in response:
            return [item['embedding'] for item in response['data']]

    # Fallback: send individually
    return [_embed_single(text) for text in texts]


@lru_cache(maxsize=1)
def _get_local_model():
    """Load and cache the local embedding model."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading local embedding model: {settings.embedding_model_name}")
    model = SentenceTransformer(
        settings.embedding_model_name,
        device=settings.embedding_device,
        trust_remote_code=True,
    )
    dim = model.get_embedding_dimension() if hasattr(model, 'get_embedding_dimension') else model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded: {dim} dimensions")
    return model


def _embed_via_local(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed using local sentence-transformers model."""
    import numpy as np
    model = _get_local_model()
    embeddings: np.ndarray = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()


def embed_texts(
    texts: list[str],
    mode: str = "passage",
    batch_size: int = 32,
) -> list[list[float]]:
    if not texts:
        return []

    prefixes = _get_prefixes()
    prefix = prefixes.get(mode, prefixes.get("passage", ""))
    prefixed = [f"{prefix}{t}" for t in texts]

    if settings.embedding_mode == "api":
        return _embed_via_api(prefixed)
    else:
        return _embed_via_local(prefixed, batch_size=batch_size)


def embed_query(query: str) -> list[float]:
    results = embed_texts([query], mode="query")
    return results[0]
