from dataclasses import dataclass

from src.ingestion.embedder import embed_query
from src.generation.llm_client import generate
from src.retrieval.models import Citation, RetrievedChunk
from src.retrieval.vector_store import VectorStore

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
        )
        for c in chunks
    ]
    return RAGResponse(answer=answer, citations=citations)


async def agent_query(question: str, user_groups: list[str], vector_store, schema_registry, metadata_store=None) -> RAGResponse:
    import asyncio
    from src.ingestion.embedder import embed_query
    from src.retrieval.query_cache import cache_lookup, cache_store

    # Check cache first
    try:
        query_vector = await asyncio.to_thread(embed_query, question)
        cached = cache_lookup(query_vector, user_groups)
        if cached:
            citations = [
                Citation(
                    doc_id=c.get("doc_id", ""), filename=c.get("filename", ""),
                    doc_type=c.get("doc_type", ""), chunk_index=c.get("chunk_index", 0),
                    page=c.get("page"), snippet=c.get("snippet", ""),
                    relevance=c.get("relevance", 0.0),
                )
                for c in cached.get("citations", [])
            ]
            return RAGResponse(answer=f"[Cached result from: \"{cached['cached_query']}\"]\n\n{cached['answer']}", citations=citations)
    except Exception:
        pass

    # Run the full agent pipeline
    from src.agent.graph import run_agent
    result = await run_agent(question=question, user_groups=user_groups, vector_store=vector_store, schema_registry=schema_registry, metadata_store=metadata_store)

    # Cache the result
    try:
        citation_dicts = [
            {"doc_id": c.doc_id, "filename": c.filename, "doc_type": c.doc_type,
             "chunk_index": c.chunk_index, "page": c.page, "snippet": c.snippet,
             "relevance": c.relevance}
            for c in result.citations
        ]
        source_ids = list({c.doc_id for c in result.citations})
        cache_store(
            query_text=question, query_vector=query_vector,
            answer=result.answer, citations=citation_dicts,
            user_groups=user_groups, source_doc_ids=source_ids,
        )
    except Exception:
        pass

    return result
