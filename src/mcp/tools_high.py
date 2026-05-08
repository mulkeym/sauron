from src.generation.rag_chain import agent_query


async def ask(
    question: str,
    user_groups: list[str],
    vector_store,
    schema_registry,
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
    format: str = "brief",
) -> dict:
    detail = "brief" if format == "brief" else "detailed"
    question = f"Provide a {detail} summary of: {topic}"

    response = await agent_query(
        question=question,
        user_groups=user_groups,
        vector_store=vector_store,
        schema_registry=schema_registry,
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
) -> dict:
    question = f"Compare and contrast: '{item_a}' vs '{item_b}'. List key differences."

    response = await agent_query(
        question=question,
        user_groups=user_groups,
        vector_store=vector_store,
        schema_registry=schema_registry,
    )

    sources = [
        {"doc_id": c.doc_id, "filename": c.filename}
        for c in response.citations
    ]

    return {"comparison": response.answer, "sources": sources}
