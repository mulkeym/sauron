"""LightRAG adapter — uses LightRAG's built-in OpenAI adapter for best format compliance."""
import asyncio
import logging
import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
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
    **kwargs,
) -> str:
    """LLM function using LightRAG's built-in OpenAI adapter.

    This adapter has better prompt construction and format enforcement
    than a raw HTTP call, resulting in more reliable entity extraction.
    """
    return await openai_complete_if_cache(
        model=settings.vllm_model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key or "not-needed",
        **kwargs,
    )


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


async def query_graph(question: str, mode: str = "hybrid") -> dict:
    """Query the knowledge graph and return a focused, synthesized answer.

    Uses LightRAG's own LLM synthesis to produce a concise, relevant
    summary instead of dumping raw graph data.
    """
    rag = await get_lightrag()
    try:
        result = await rag.aquery(
            question,
            param=QueryParam(
                mode=mode,
                only_need_context=False,  # let LightRAG synthesize a focused answer
                top_k=20,
                response_type="Brief bullet points focusing on specific names, amounts, and relationships",
            ),
        )

        if not result or len(result.strip()) < 20:
            return {"context": "", "mode": mode}

        return {"context": result.strip(), "mode": mode}
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
