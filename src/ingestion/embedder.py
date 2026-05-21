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
    """Load and cache the local embedding model."""
    import os
    from sentence_transformers import SentenceTransformer

    device = settings.embedding_device
    if device == "multi-gpu":
        device = "cuda"  # base model on first GPU; multi-GPU handled in encode

    # Maximize CPU thread usage
    if device == "cpu":
        import torch
        cores = os.cpu_count() or 4
        torch.set_num_threads(cores)
        try:
            torch.set_num_interop_threads(cores)
        except RuntimeError:
            pass  # already set or parallel work started
        logger.info(f"CPU threading: {cores} cores")

    logger.info(f"Loading local embedding model: {settings.embedding_model_name} on {device}")
    model = SentenceTransformer(
        settings.embedding_model_name,
        device=device,
        trust_remote_code=True,
    )
    dim = model.get_embedding_dimension() if hasattr(model, 'get_embedding_dimension') else model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded: {dim} dimensions on {device}")
    return model


def _get_gpu_count() -> int:
    """Detect available CUDA GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _embed_via_local(texts: list[str], batch_size: int = 0) -> list[list[float]]:
    """Embed using local sentence-transformers model."""
    import numpy as np
    if batch_size == 0:
        batch_size = settings.embedding_batch_size

    model = _get_local_model()

    # Multi-GPU: use encode_multi_process for parallel encoding
    if settings.embedding_device == "multi-gpu":
        gpu_count = _get_gpu_count()
        if gpu_count > 1:
            pool = model.start_multi_process_pool(
                target_devices=[f"cuda:{i}" for i in range(gpu_count)]
            )
            logger.info(f"Multi-GPU embedding: {len(texts)} texts across {gpu_count} GPUs")
            embeddings = model.encode_multi_process(texts, pool, batch_size=batch_size)
            model.stop_multi_process_pool(pool)
            return embeddings.tolist()

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
