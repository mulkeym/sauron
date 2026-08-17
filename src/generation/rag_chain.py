from dataclasses import dataclass

from src.ingestion.embedder import embed_query
from src.generation.llm_client import generate
from src.retrieval.models import Citation, RetrievedChunk
from src.retrieval.vector_store import VectorStore
from src.retrieval.query_cache import judged_cache_lookup, cache_store

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on provided context documents.
Rules:
- Only answer based on the provided context. Do not use outside knowledge.
- Cite your sources using [N] notation, where N corresponds to the context chunk number.
- If the context does not contain enough information to answer, say so clearly.
- Be concise and accurate."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer the question based only on the context above. Cite sources using [N] notation."""


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]
    cached: bool = False
    cached_query: str | None = None
    query_type: str = ""


def rag_query(question, user_groups, vector_store, top_k=10):
    query_vector = embed_query(question)
    chunks = vector_store.search(vector=query_vector, user_groups=user_groups, top_k=top_k)
    if not chunks:
        return RAGResponse(
            answer="I could not find any relevant information in the documents you have access to.",
            citations=[],
        )
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[{i}] {chunk.metadata.filename}"
        if chunk.metadata.page is not None:
            source += f", page {chunk.metadata.page}"
        if chunk.metadata.figure_id:
            source += f", figure {chunk.metadata.figure_id}"
        if chunk.metadata.slide is not None:
            source += f", slide {chunk.metadata.slide}"
        context_parts.append(f"{source}:\n{chunk.text}")
    context = "\n\n".join(context_parts)
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    answer = generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    citations = [
        Citation(
            doc_id=c.metadata.doc_id,
            filename=c.metadata.filename,
            doc_type=c.metadata.doc_type,
            chunk_index=c.metadata.chunk_index,
            page=c.metadata.page,
            snippet=c.text[:200],
            relevance=c.score,
            figure_id=c.metadata.figure_id,
            section_title=c.metadata.section_title,
            caption=c.metadata.caption,
            slide=c.metadata.slide,
        )
        for c in chunks
    ]
    return RAGResponse(answer=answer, citations=citations)


async def agent_query_streamed(
    question: str, user_groups: list[str], vector_store, schema_registry,
    metadata_store=None, step_callback=None, skip_cache: bool = False,
    session_headers=None, agent_id: str | None = None, session_id: str | None = None,
) -> RAGResponse:
    from src.generation.llm_client import llm_session
    with llm_session(headers=session_headers, agent_id=agent_id, session_id=session_id):
        return await _agent_query_streamed_bound(
            question=question, user_groups=user_groups, vector_store=vector_store,
            schema_registry=schema_registry, metadata_store=metadata_store,
            step_callback=step_callback, skip_cache=skip_cache,
        )


async def _agent_query_streamed_bound(
    question: str, user_groups: list[str], vector_store, schema_registry,
    metadata_store=None, step_callback=None, skip_cache: bool = False,
) -> RAGResponse:
    # Surface the cache lookup as the first observable step. It runs before the
    # graph, so it is not a graph node — emit it explicitly (the spec's data flow
    # lists "checking cache" as a step callers should see).
    if step_callback is not None:
        step_callback("cache_check")
    # Shared cache decision (embed -> lookup -> LLM applicability judge) — same
    # path the admin playground uses, so the two cannot diverge.
    decision = await judged_cache_lookup(question, user_groups, skip_cache=skip_cache)
    if decision.accepted:
        cached = decision.cached
        citations = [
            Citation(
                doc_id=c.get("doc_id", ""), filename=c.get("filename", ""),
                doc_type=c.get("doc_type", ""), chunk_index=c.get("chunk_index", 0),
                page=c.get("page"), snippet=c.get("snippet", ""),
                relevance=c.get("relevance", 0.0),
                figure_id=c.get("figure_id"), section_title=c.get("section_title"),
                caption=c.get("caption"),
                slide=c.get("slide"),
            )
            for c in cached.get("citations", [])
        ]
        return RAGResponse(
            answer=cached["answer"], citations=citations,
            cached=True, cached_query=cached.get("cached_query"),
            query_type="cache",
        )

    # run_agent_streamed is imported lazily to avoid a circular import: src.agent.graph
    # imports RAGResponse from this module. Tests patch src.agent.graph.run_agent_streamed.
    from src.agent.graph import run_agent_streamed
    result = await run_agent_streamed(
        question=question, user_groups=user_groups, vector_store=vector_store,
        schema_registry=schema_registry, metadata_store=metadata_store,
        step_callback=step_callback,
    )

    if decision.query_vector is not None:
        try:
            citation_dicts = [
                {"doc_id": c.doc_id, "filename": c.filename, "doc_type": c.doc_type,
                 "chunk_index": c.chunk_index, "page": c.page, "snippet": c.snippet,
                 "relevance": c.relevance, "figure_id": c.figure_id,
                 "section_title": c.section_title, "caption": c.caption,
                 "slide": c.slide}
                for c in result.citations
            ]
            source_ids = list({c.doc_id for c in result.citations})
            cache_store(
                query_text=question, query_vector=decision.query_vector,
                answer=result.answer, citations=citation_dicts,
                user_groups=user_groups, source_doc_ids=source_ids,
            )
        except Exception:
            pass

    return result


async def agent_query(
    question: str, user_groups: list[str], vector_store, schema_registry,
    metadata_store=None, skip_cache: bool = False,
    session_headers=None, agent_id: str | None = None, session_id: str | None = None,
) -> RAGResponse:
    return await agent_query_streamed(
        question=question, user_groups=user_groups, vector_store=vector_store,
        schema_registry=schema_registry, metadata_store=metadata_store,
        step_callback=None, skip_cache=skip_cache,
        session_headers=session_headers, agent_id=agent_id, session_id=session_id,
    )
