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
        llm_model_max_async=4,

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


def _get_allowed_filenames(user_groups: list[str]) -> set[str] | None:
    """Get filenames the user can access based on ACL groups.
    Returns None if user has ALL access (no filtering needed).
    """
    if "ALL" in user_groups:
        return None  # no filtering

    import asyncio
    from src.db.metadata import MetadataStore

    async def _fetch():
        store = MetadataStore()
        await store.init()
        docs = await store.list_documents(user_groups)
        return {d.filename for d in docs}

    try:
        return asyncio.run(_fetch())
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _fetch()).result()


def _files_to_allowed_entities(allowed_files: set[str]) -> set[str]:
    """Trace filenames through LightRAG chunks to find entity names."""
    import json as json_mod
    import re
    from pathlib import Path

    chunks_file = Path("data/lightrag/kv_store_text_chunks.json")
    if not chunks_file.exists():
        return set()

    chunk_data = json_mod.loads(chunks_file.read_text())
    allowed_chunks = {cid for cid, data in chunk_data.items()
                      if data.get("file_path", "") in allowed_files}

    graphml = Path("data/lightrag/graph_chunk_entity_relation.graphml")
    if not graphml.exists():
        return set()

    content = graphml.read_text()
    allowed_entities = set()

    for match in re.finditer(
        r'<node id="([^"]+)"[^>]*>(.*?)</node>', content, re.DOTALL
    ):
        name = match.group(1)
        source_match = re.search(r'<data key="d3">(.*?)</data>', match.group(2), re.DOTALL)
        if source_match:
            source_chunks = source_match.group(1).replace("&lt;SEP&gt;", "<SEP>").split("<SEP>")
            if any(c.strip() in allowed_chunks for c in source_chunks):
                allowed_entities.add(name)

    return allowed_entities


def _get_acl_allowed_entities(user_groups: list[str]) -> set[str] | None:
    """Get entity names the user can see based on source document ACLs.
    Returns None if user has ALL access.
    """
    if "ALL" in user_groups:
        return None

    allowed_files = _get_allowed_filenames(user_groups)
    if allowed_files is None:
        return None

    return _files_to_allowed_entities(allowed_files)


def _get_app_allowed_entities(app_id: int) -> set[str] | None:
    """Get entity names from documents belonging to a specific application.
    Returns None if app_id is 0 (no filtering).
    """
    if not app_id:
        return None

    import asyncio
    from src.db.metadata import MetadataStore

    async def _fetch():
        store = MetadataStore()
        await store.init()
        docs = await store.list_documents()
        return {d.filename for d in docs if d.application_id == app_id}

    try:
        allowed_files = asyncio.run(_fetch())
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            allowed_files = pool.submit(asyncio.run, _fetch()).result()

    if not allowed_files:
        return set()

    return _files_to_allowed_entities(allowed_files)


async def query_graph(question: str, mode: str = "hybrid", user_groups: list[str] | None = None) -> dict:
    """Query the knowledge graph with ACL filtering.

    If user_groups is provided and doesn't contain "ALL", only returns
    context from documents the user can access.
    """
    rag = await get_lightrag()

    # Reduce top_k for non-admin users to speed up queries
    top_k = 20
    if user_groups and "ALL" not in user_groups:
        top_k = 10  # fewer entities to search = faster

    try:
        result = await rag.aquery(
            question,
            param=QueryParam(
                mode=mode,
                only_need_context=False,
                top_k=top_k,
                response_type="Brief bullet points focusing on specific names, amounts, and relationships",
            ),
        )

        if not result or len(result.strip()) < 20:
            return {"context": "", "mode": mode}

        # ACL post-filter: if user doesn't have ALL access, verify the
        # response only references documents they can see
        if user_groups and "ALL" not in user_groups:
            allowed = _get_acl_allowed_entities(user_groups)
            if allowed is not None:
                # Filter lines that reference entities the user can't see
                filtered_lines = []
                for line in result.strip().split("\n"):
                    # Keep lines that reference allowed entities or are generic
                    line_lower = line.lower()
                    if any(ent.lower() in line_lower for ent in allowed) or not line.strip().startswith("-"):
                        filtered_lines.append(line)
                result = "\n".join(filtered_lines)

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
