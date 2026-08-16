from __future__ import annotations
from src.generation.llm_client import generate
from src.generation.rag_chain import agent_query
from src.mcp.auth import mcp_llm_session_kwargs
from src.mcp.tools_low import list_documents_in_category, lookup_document


async def ask(
    question: str,
    user_groups: list[str],
    vector_store,
    schema_registry,
    metadata_store=None,
    depth: str = "thorough",
    context: str | None = None,
) -> dict:
    if context:
        full_question = f"Context: {context}\n\nQuestion: {question}"
    else:
        full_question = question

    response = await agent_query(
        question=full_question,
        user_groups=user_groups,
        vector_store=vector_store,
        schema_registry=schema_registry,
        metadata_store=metadata_store,
        **mcp_llm_session_kwargs(),
    )

    citations = [
        {
            "doc_id": c.doc_id,
            "filename": c.filename,
            "doc_type": c.doc_type,
            "chunk_index": c.chunk_index,
            "page": c.page,
            "snippet": c.snippet,
            "relevance": c.relevance,
            "figure_id": c.figure_id,
            "section_title": c.section_title,
            "caption": c.caption,
            "slide": c.slide,
        }
        for c in response.citations
    ]

    return {
        "answer": response.answer,
        "citations": citations,
        "retrieval_strategy": depth,
        "_call_args": {"question": full_question},
    }


async def summarize_topic(
    topic: str,
    user_groups: list[str],
    vector_store,
    schema_registry,
    metadata_store=None,
    format: str = "brief",
) -> dict:
    detail = "brief" if format == "brief" else "detailed"
    question = f"Provide a {detail} summary of: {topic}"

    response = await agent_query(
        question=question,
        user_groups=user_groups,
        vector_store=vector_store,
        schema_registry=schema_registry,
        metadata_store=metadata_store,
        **mcp_llm_session_kwargs(),
    )

    sources = [
        {"doc_id": c.doc_id, "filename": c.filename}
        for c in response.citations
    ]

    return {"summary": response.answer, "sources": sources}


async def compare(
    item_a: str,
    item_b: str,
    user_groups: list[str],
    vector_store,
    schema_registry,
    metadata_store=None,
) -> dict:
    question = f"Compare and contrast: '{item_a}' vs '{item_b}'. List key differences."

    response = await agent_query(
        question=question,
        user_groups=user_groups,
        vector_store=vector_store,
        schema_registry=schema_registry,
        metadata_store=metadata_store,
        **mcp_llm_session_kwargs(),
    )

    sources = [
        {"doc_id": c.doc_id, "filename": c.filename}
        for c in response.citations
    ]

    return {"comparison": response.answer, "sources": sources}


def summarize_documents(
    category: str,
    user_groups: list[str],
    vector_store,
    metadata_store,
) -> dict:
    """List all docs in a category, read each one, and return a summary per document."""
    docs = list_documents_in_category(
        category=category, user_groups=user_groups, metadata_store=metadata_store,
    )
    if not docs:
        # If no category matched, list only documents visible to this caller.
        import asyncio
        filter_groups = None if "ALL" in user_groups else user_groups
        try:
            all_docs = asyncio.run(metadata_store.list_documents(filter_groups))
        except RuntimeError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                all_docs = pool.submit(
                    asyncio.run, metadata_store.list_documents(filter_groups)
                ).result()
        docs = [{"doc_id": d.doc_id, "filename": d.filename, "doc_type": d.doc_type, "category": d.category or "uncategorized"} for d in all_docs]

    summaries = []
    for doc in docs:
        content_result = lookup_document(
            doc_id=doc["doc_id"], user_groups=user_groups, vector_store=vector_store,
        )
        content = content_result.get("content", "")
        if not content:
            summaries.append({"filename": doc["filename"], "summary": "Could not read document content."})
            continue

        summary = generate(
            system_prompt="Summarize the following document in 2-3 sentences.",
            user_prompt=content[:3000],
            max_tokens=2048,
        )
        summaries.append({"filename": doc["filename"], "doc_id": doc["doc_id"], "doc_type": doc.get("doc_type", ""), "summary": summary})

    return {"category": category, "document_count": len(summaries), "summaries": summaries}
