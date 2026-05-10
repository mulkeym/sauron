import json
import subprocess
from functools import lru_cache

from src.config import settings

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


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
        timeout=60
    )

    if result.returncode != 0:
        raise RuntimeError(f"Embedding request failed: {result.stderr}")

    response = json.loads(result.stdout)
    if 'error' in response:
        raise RuntimeError(f"Embedding error: {response['error']}")

    return [item['embedding'] for item in response['data']]


def _embed_via_local(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Load model locally via sentence-transformers."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    @lru_cache(maxsize=1)
    def _get_model():
        return SentenceTransformer(settings.embedding_model_name, device=settings.embedding_device)

    model = _get_model()
    embeddings: np.ndarray = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()


def embed_texts(
    texts: list[str],
    prefix: str = PASSAGE_PREFIX,
    batch_size: int = 32,
) -> list[list[float]]:
    if not texts:
        return []

    prefixed = [f"{prefix}{t}" for t in texts]

    if settings.embedding_mode == "api":
        return _embed_via_api(prefixed)
    else:
        return _embed_via_local(prefixed, batch_size=batch_size)


def embed_query(query: str) -> list[float]:
    results = embed_texts([query], prefix=QUERY_PREFIX)
    return results[0]
