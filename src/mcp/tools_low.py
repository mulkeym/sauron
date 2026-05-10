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
) -> list[dict]:
    vector = embed_query(query)
    chunks = vector_store.search(vector=vector, user_groups=user_groups, top_k=top_k)
    # Only filter by doc_type if it's a known type (pdf, docx, xlsx, transcript)
    valid_types = {"pdf", "docx", "xlsx", "transcript", "txt", "markdown"}
    if doc_type and doc_type.lower() in valid_types:
        chunks = [c for c in chunks if c.metadata.doc_type == doc_type.lower()]
    results = []
    for chunk in chunks:
        results.append(
            {
                "text": chunk.text,
                "source": chunk.metadata.filename,
                "doc_id": chunk.metadata.doc_id,
                "doc_type": chunk.metadata.doc_type,
                "page": chunk.metadata.page,
                "relevance": chunk.score,
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
            return {"sql": "", "results": [], "answer": result.get("answer", ""), "citations": result.get("citations", [])}
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


def lookup_document(
    doc_id: str,
    user_groups: list[str],
    vector_store,
) -> dict:
    vector = embed_query(f"document {doc_id}")
    chunks = vector_store.search(vector=vector, user_groups=user_groups, top_k=100)
    # Match by doc_id (UUID) or by filename
    matching = [c for c in chunks if c.metadata.doc_id == doc_id or c.metadata.filename == doc_id]
    if not matching:
        return {"content": "", "metadata": {}, "error": f"Document '{doc_id}' not found. Use tool_list_documents to get valid doc_ids or filenames."}
    matching_sorted = sorted(matching, key=lambda c: c.metadata.chunk_index)
    content = "\n".join(c.text for c in matching_sorted)
    first_meta = matching_sorted[0].metadata
    metadata = {
        "doc_id": first_meta.doc_id,
        "filename": first_meta.filename,
        "doc_type": first_meta.doc_type,
        "category": first_meta.category,
        "acl_groups": first_meta.acl_groups,
    }
    return {"content": content, "metadata": metadata}


def search_meetings(
    user_groups: list[str],
    vector_store,
    topic: str | None = None,
    speaker: str | None = None,
    type_filter: str | None = None,
    top_k: int = 50,
) -> list[dict]:
    query = topic if topic else "meeting transcript"
    vector = embed_query(query)
    chunks = vector_store.search(vector=vector, user_groups=user_groups, top_k=top_k)
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


async def search_knowledge_graph(query, metadata_store, entity_type=None):
    entities = await metadata_store.search_entities(query, entity_type=entity_type)
    if not entities:
        return {"entity": None, "error": f"No entity found matching '{query}'", "suggestions": "Try a different name or use tool_list_documents to find documents first."}
    best = entities[0]
    details = await metadata_store.get_entity_details(best.id)
    other_matches = [{"name": e.name, "type": e.entity_type} for e in entities[1:5]]
    return {"entity": details["entity"], "mentions_in": details["mentions"], "relationships": details["relationships"], "other_matches": other_matches}


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
