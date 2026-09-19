from __future__ import annotations
import asyncio
import re

from src.ingestion.embedder import embed_query
from src.generation.llm_client import generate
from src.db.sql_executor import execute_sql
from src.config import settings


SQL_SYSTEM_PROMPT = (
    "You are a SQL expert. Given a schema, write a single valid SELECT query "
    "that answers the user's question. Return only the SQL, no explanation."
)


def search_documents(
    query: str,
    user_groups: list[str],
    vector_store,
    doc_type: str | None = None,
    top_k: int = 10,
    metadata_store=None,
) -> list[dict]:
    from src.retrieval.query_scope import resolve_query_scope_sync
    scope = resolve_query_scope_sync(user_groups, metadata_store)
    if not scope.doc_ids:
        return []
    vector = embed_query(query)
    chunks = vector_store.search(vector=vector, user_groups=user_groups, top_k=top_k,
                                 doc_ids=list(scope.doc_ids))
    # Only filter by doc_type if it's a known type (pdf, docx, xlsx, transcript)
    valid_types = {"pdf", "docx", "xlsx", "transcript", "txt", "markdown"}
    if doc_type and doc_type.lower() in valid_types:
        chunks = [c for c in chunks if c.metadata.doc_type == doc_type.lower()]
    async def stored_figures():
        return await metadata_store.list_figures(list(scope.doc_ids)) if metadata_store else []
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        records = asyncio.run(stored_figures())
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            records = pool.submit(asyncio.run, stored_figures()).result()
    available = {(r["doc_id"], r["figure_id"]) for r in records if r.get("assets")}
    results = []
    for chunk in chunks:
        results.append(
            {
                "image_available": (chunk.metadata.doc_id, chunk.metadata.figure_id) in available,
                "text": chunk.text,
                "source": chunk.metadata.filename,
                "doc_id": chunk.metadata.doc_id,
                "doc_type": chunk.metadata.doc_type,
                "page": chunk.metadata.page,
                "relevance": chunk.score,
                "content_type": chunk.metadata.content_type,
                "figure_id": chunk.metadata.figure_id,
                "figure_kind": chunk.metadata.figure_kind,
                "caption": chunk.metadata.caption,
                "slide": chunk.metadata.slide,
                "source_locator": chunk.metadata.source_locator,
            }
        )
    return results


async def query_database(
    question: str,
    user_groups: list[str],
    schema_registry,
    vector_store=None,
    metadata_store=None,
) -> dict:
    schema_prompt = schema_registry.schemas_to_prompt(user_groups)
    if schema_prompt == "No database schemas available." or not schema_registry.list_for_user(user_groups):
        # No database schemas — fall back to document search via the RAG agent
        if vector_store:
            from src.mcp.tools_high import ask
            result = await ask(
                question=question,
                user_groups=user_groups,
                vector_store=vector_store,
                schema_registry=schema_registry,
                metadata_store=metadata_store,
            )
            return {
                "sql": "",
                "results": [],
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "query_type": result.get("query_type", ""),
                "cached": bool(result.get("cached")),
            }
        return {"sql": "", "results": [], "error": "No database schemas available for your groups."}

    user_prompt = f"Schema:\n{schema_prompt}\n\nQuestion: {question}"
    sql_raw = generate(system_prompt=SQL_SYSTEM_PROMPT, user_prompt=user_prompt)

    # Strip markdown code fences if present
    sql = re.sub(r"```(?:sql)?\s*", "", sql_raw, flags=re.IGNORECASE).replace("```", "").strip()

    # Determine which database to run against
    schemas = schema_registry.list_for_user(user_groups)
    db_name = schemas[0].database if schemas else None
    db_registry = settings.database_registry
    database_url = db_registry.get(db_name, "") if db_name else ""

    try:
        rows = await execute_sql(database_url, sql)
        return {"sql": sql, "results": rows}
    except Exception as exc:
        return {"sql": sql, "results": [], "error": str(exc)}


def lookup_document(doc_id: str, user_groups: list[str], vector_store,
                    metadata_store=None, offset: int = 0, limit: int = 100) -> dict:
    """Read a page of authorized indexed passages by exact ID or unique filename."""
    if offset < 0 or not 1 <= limit <= 200:
        return {"error": "offset must be nonnegative and limit must be 1–200.", "content": "", "metadata": {}}
    if not user_groups:
        return {"content": "", "metadata": {}, "error": "Document not found or not accessible."}
    if metadata_store is None:
        from src.api.routes_ingest import get_metadata_store
        metadata_store = get_metadata_store()
    async def resolve():
        docs = await metadata_store.list_documents(None if "ALL" in user_groups else user_groups)
        # Check again at this boundary so even a stale vector ACL cannot grant access.
        docs = [d for d in docs if "ALL" in user_groups or set(user_groups).intersection(d.acl_groups)]
        exact = [d for d in docs if d.doc_id == doc_id]
        return exact or [d for d in docs if d.filename == doc_id]
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        matching = asyncio.run(resolve())
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            matching = pool.submit(asyncio.run, resolve()).result()
    if len(matching) != 1:
        error = "Filename is ambiguous; use a document ID." if matching else "Document not found or not accessible."
        return {"content": "", "metadata": {}, "error": error}
    doc = matching[0]
    chunks, more = vector_store.read_document_page(doc.doc_id, user_groups, offset=offset, limit=limit)
    # The cursor follows index order; explicit source positions let clients
    # reconstruct reading order without assuming chunks are contiguous prose.
    content = "\n\n".join(c.text for c in chunks)
    return {"content": content,
            "metadata": {"doc_id": doc.doc_id, "filename": doc.filename,
                         "doc_type": doc.doc_type, "category": doc.category,
                         "source_url": getattr(doc, "source_url", "") or ""},
            "chunks": [c.model_dump() for c in chunks], "offset": offset,
            "next_offset": offset + len(chunks) if more else None,
            "complete": offset == 0 and not more,
            "representation": "indexed_passages"}


def search_meetings(
    user_groups: list[str],
    vector_store,
    topic: str | None = None,
    speaker: str | None = None,
    type_filter: str | None = None,
    top_k: int = 50,
    metadata_store=None,
) -> list[dict]:
    from src.retrieval.query_scope import resolve_query_scope_sync
    scope = resolve_query_scope_sync(user_groups, metadata_store)
    if not scope.doc_ids:
        return []
    query = topic if topic else "meeting transcript"
    vector = embed_query(query)
    chunks = vector_store.search(vector=vector, user_groups=user_groups, top_k=top_k,
                                 doc_ids=list(scope.doc_ids))
    # Filter to transcripts only
    chunks = [c for c in chunks if c.metadata.doc_type == "transcript"]
    if speaker is not None:
        chunks = [c for c in chunks if c.metadata.speaker == speaker]
    if type_filter is not None:
        chunks = [c for c in chunks if c.metadata.utterance_type == type_filter]
    results = []
    for chunk in chunks:
        results.append(
            {
                "text": chunk.text,
                "speaker": chunk.metadata.speaker,
                "meeting": chunk.metadata.filename,
                "type": chunk.metadata.utterance_type,
                "relevance": chunk.score,
                "content_type": chunk.metadata.content_type,
                "figure_id": chunk.metadata.figure_id,
                "figure_kind": chunk.metadata.figure_kind,
                "caption": chunk.metadata.caption,
                "slide": chunk.metadata.slide,
                "source_locator": chunk.metadata.source_locator,
            }
        )
    return results


def list_documents_in_category(
    category: str,
    user_groups: list[str],
    metadata_store,
) -> list[dict]:
    filter_groups = None if "ALL" in user_groups else user_groups
    try:
        docs = asyncio.run(metadata_store.list_documents(filter_groups))
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, metadata_store.list_documents(filter_groups))
            docs = future.result()

    matching = [d for d in docs if (d.category or "uncategorized") == category]
    return [
        {
            "doc_id": d.doc_id,
            "filename": d.filename,
            "doc_type": d.doc_type,
            "category": d.category or "uncategorized",
            "chunk_count": d.chunk_count,
            "uploaded_by": d.uploaded_by,
        }
        for d in matching
    ]


async def search_knowledge_graph(
    query,
    user_groups: list[str],
    metadata_store=None,
    entity_type=None,
):
    """Search KG-derived context while enforcing source-document ACLs."""
    from src.knowledge.graph_rag import query_graph
    try:
        result = await query_graph(query, mode="local", user_groups=user_groups)
        result["query"] = query
        if entity_type:
            result["entity_type"] = entity_type
        return result
    except Exception as e:
        return {"query": query, "error": str(e)}


def list_sources(
    user_groups: list[str],
    metadata_store,
) -> list[dict]:
    # Pass None to list_documents when ALL access (skip ACL filtering)
    filter_groups = None if "ALL" in user_groups else user_groups
    try:
        docs = asyncio.run(metadata_store.list_documents(filter_groups))
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run, metadata_store.list_documents(filter_groups)
            )
            docs = future.result()

    # Group by category
    groups: dict[str, list] = {}
    for doc in docs:
        cat = getattr(doc, "category", "") or "uncategorized"
        groups.setdefault(cat, []).append(doc)

    results = []
    for category, group_docs in groups.items():
        doc_types = list({d.doc_type for d in group_docs})
        results.append(
            {
                "name": category,
                "type": doc_types[0] if len(doc_types) == 1 else "mixed",
                "doc_count": len(group_docs),
            }
        )
    return results
