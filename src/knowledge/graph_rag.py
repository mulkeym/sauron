"""LightRAG adapter — wraps our LLM and embedding functions for LightRAG's API."""
import asyncio
import logging
import numpy as np

import aiohttp

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from src.config import settings

logger = logging.getLogger(__name__)

_rag_instance: LightRAG | None = None
_initialized = False


async def _llm_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] = [],
    keyword_extraction: bool = False,
    enable_cot: bool = False,
    stream: bool = False,
    **kwargs,
) -> str:
    """LLM function for LightRAG — async HTTP to avoid blocking the event loop."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": settings.vllm_model_name,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0 if keyword_extraction else 0.1,
    }

    try:
        timeout = aiohttp.ClientTimeout(total=settings.vllm_request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f'{settings.vllm_base_url}/chat/completions',
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        message = data["choices"][0]["message"]
        content = message.get("content", "").strip()

        # Fallback for thinking models
        if not content:
            content = (message.get("reasoning_content") or message.get("reasoning") or "").strip()

        return content
    except Exception as e:
        logger.error(f"LightRAG LLM call failed: {e}")
        return ""


async def _embed_func(texts: list[str], **kwargs) -> np.ndarray:
    """Embedding function for LightRAG — runs in thread to avoid blocking."""
    from src.ingestion.embedder import embed_texts
    vectors = await asyncio.to_thread(embed_texts, texts, "passage")
    return np.array(vectors)


def _detect_embed_dim() -> int:
    """Detect embedding dimension."""
    if settings.embedding_dimension > 0:
        return settings.embedding_dimension
    from src.ingestion.embedder import embed_texts
    vectors = embed_texts(["dimension probe"], mode="passage")
    dim = len(vectors[0])
    settings.embedding_dimension = dim
    return dim


async def get_lightrag() -> LightRAG:
    """Get or create the LightRAG instance."""
    global _rag_instance, _initialized

    if _rag_instance is not None and _initialized:
        return _rag_instance

    embed_dim = _detect_embed_dim()
    logger.info(f"Initializing LightRAG (embedding dim: {embed_dim})")

    _rag_instance = LightRAG(
        working_dir="data/lightrag",
        llm_model_func=_llm_func,
        llm_model_name=settings.vllm_model_name,
        llm_model_max_async=2,

        embedding_func=EmbeddingFunc(
            embedding_dim=embed_dim,
            max_token_size=8192,
            func=_embed_func,
        ),
        embedding_batch_num=10,

        # Chunking — smaller chunks help smaller models extract entities more reliably
        chunk_token_size=500,
        chunk_overlap_token_size=50,

        # Storage — file-based, zero infrastructure
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",

        # Retrieval
        top_k=30,
        chunk_top_k=5,
        enable_llm_cache=True,
    )

    await _rag_instance.initialize_storages()
    _initialized = True
    logger.info("LightRAG initialized successfully")
    return _rag_instance


async def insert_document(text: str, doc_id: str = "", filename: str = "") -> str:
    """Insert a document into LightRAG for knowledge graph extraction."""
    rag = await get_lightrag()
    try:
        result = await rag.ainsert(
            text,
            ids=[doc_id] if doc_id else None,
            file_paths=[filename] if filename else None,
        )
        logger.info(f"LightRAG insert complete: {filename or doc_id}")
        return result
    except Exception as e:
        logger.error(f"LightRAG insert failed: {e}")
        return f"error: {e}"


async def query_graph(question: str, mode: str = "mix") -> dict:
    """Query the knowledge graph.

    Modes:
    - local: entities + relationships (specific entity questions)
    - global: high-level patterns and themes
    - hybrid: local + global combined
    - mix: knowledge graph + vector chunks (recommended)
    - naive: vector search only (no graph)
    """
    rag = await get_lightrag()
    try:
        result = await rag.aquery(
            question,
            param=QueryParam(
                mode=mode,
                only_need_context=True,  # return context, we'll synthesize ourselves
            ),
        )
        return {"context": result, "mode": mode}
    except Exception as e:
        logger.error(f"LightRAG query failed: {e}")
        return {"context": "", "mode": mode, "error": str(e)}


async def get_graph_data(question: str, mode: str = "local") -> dict:
    """Get raw graph data (entities, relationships) without LLM synthesis."""
    rag = await get_lightrag()
    try:
        data = await rag.aquery_data(question, param=QueryParam(mode=mode))
        return data
    except Exception as e:
        logger.error(f"LightRAG query_data failed: {e}")
        return {}


async def get_knowledge_graph(entity_name: str, max_depth: int = 3) -> dict:
    """Get a subgraph starting from an entity."""
    rag = await get_lightrag()
    try:
        kg = await rag.get_knowledge_graph(
            node_label=entity_name,
            max_depth=max_depth,
            max_nodes=100,
        )
        return kg
    except Exception as e:
        logger.error(f"LightRAG get_knowledge_graph failed: {e}")
        return {}


async def is_graph_populated() -> bool:
    """Check if the knowledge graph has any data."""
    rag = await get_lightrag()
    try:
        labels = await rag.get_graph_labels()
        return bool(labels)
    except Exception:
        return False
