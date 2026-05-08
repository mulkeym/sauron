from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name, device=settings.embedding_device)


def embed_texts(texts: list[str], prefix: str = PASSAGE_PREFIX, batch_size: int = 32) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    prefixed = [f"{prefix}{t}" for t in texts]
    embeddings: np.ndarray = model.encode(prefixed, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    results = embed_texts([query], prefix=QUERY_PREFIX)
    return results[0]
