import json
import logging
from functools import lru_cache

import requests

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


def _embed_via_api(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Call an OpenAI-compatible /v1/embeddings endpoint in batches."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = requests.post(
                f'{settings.embedding_api_url}/embeddings',
                json={"model": settings.embedding_model_name, "input": batch},
                timeout=120,
                verify=settings.ssl_verify,
            )
            resp.raise_for_status()
            data = resp.json()
            if 'data' in data:
                all_embeddings.extend(item['embedding'] for item in data['data'])
                continue
        except Exception as e:
            logger.warning(f"Batch embedding failed (batch {i//batch_size + 1}), trying individually: {e}")

        # Fallback: send individually for this batch
        for text in batch:
            try:
                resp = requests.post(
                    f'{settings.embedding_api_url}/embeddings',
                    json={"model": settings.embedding_model_name, "input": text},
                    timeout=60,
                    verify=settings.ssl_verify,
                )
                resp.raise_for_status()
                all_embeddings.append(resp.json()['data'][0]['embedding'])
            except Exception as e:
                logger.error(f"Individual embedding failed: {e}")
                raise

    return all_embeddings


@lru_cache(maxsize=1)
def _get_local_model():
    """Load and cache the local embedding model (CPU-only, no GPU required)."""
    import os
    from sentence_transformers import SentenceTransformer

    # Maximize CPU thread usage
    import torch
    cores = os.cpu_count() or 4
    torch.set_num_threads(cores)
    try:
        torch.set_num_interop_threads(cores)
    except RuntimeError:
        pass  # already set or parallel work started
    logger.info(f"CPU threading: {cores} cores")

    # Prefer baked HF cache (image build). Offline / local_files_only avoids
    # runtime downloads of nomic-ai/* (and nomic-bert-2048 remote code).
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip() in ("1", "true", "True")
    offline = offline or os.environ.get("TRANSFORMERS_OFFLINE", "").strip() in ("1", "true", "True")
    logger.info(
        f"Loading local embedding model: {settings.embedding_model_name} on cpu "
        f"(local_files_only={offline})"
    )
    model = SentenceTransformer(
        settings.embedding_model_name,
        device="cpu",
        trust_remote_code=True,
        local_files_only=offline,
    )
    dim = model.get_embedding_dimension() if hasattr(model, 'get_embedding_dimension') else model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded: {dim} dimensions on cpu")
    return model


def _embed_via_local(texts: list[str], batch_size: int = 0) -> list[list[float]]:
    """Embed using local sentence-transformers model on CPU."""
    import numpy as np
    if batch_size == 0:
        batch_size = settings.embedding_batch_size

    model = _get_local_model()
    embeddings: np.ndarray = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()


def _get_max_embed_chars() -> int:
    """Return safe max chars based on the embedding model's token limit.

    Token-to-char ratio is ~4 chars/token for English text.
    """
    model = settings.embedding_model_name.lower()
    if "nomic" in model:
        return 8000   # nomic: 8192 token limit
    elif "e5" in model:
        return 1800   # e5: 512 token limit
    return 2000       # conservative default for unknown models


def _truncate_for_embedding(texts: list[str]) -> list[str]:
    """Truncate texts to fit within the embedding model's token limit."""
    max_chars = _get_max_embed_chars()
    return [t[:max_chars] if len(t) > max_chars else t for t in texts]


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
    prefixed = _truncate_for_embedding(prefixed)

    if settings.embedding_mode == "api":
        return _embed_via_api(prefixed)
    else:
        return _embed_via_local(prefixed, batch_size=batch_size)


def embed_query(query: str) -> list[float]:
    results = embed_texts([query], mode="query")
    return results[0]
