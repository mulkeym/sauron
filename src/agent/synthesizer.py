import json
import logging
from src.agent.state import AgentState
from src.generation.llm_client import generate, parse_json_response
from src.retrieval.models import Citation

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on provided context.

Rules:
- Only answer based on the provided context. Do not use outside knowledge.
- Cite document sources using [N] notation, where N corresponds to the context chunk number.
- If SQL results are provided, reference them in your answer.
- If the context does not contain enough information, say so clearly.
- Be THOROUGH and COMPLETE. Include ALL relevant information from the context, not just the first match.
- When asked about what someone said or asked, list EVERY instance found in the context.
- When listing items, use bullet points or numbered lists for clarity."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer the question thoroughly based on ALL the context above. Include every relevant detail found. Cite sources using [N] notation."""

RELEVANCE_PROMPT = """Rate each chunk's relevance to the question. Return ONLY a JSON array of chunk numbers (1-based) that are relevant.

Question: {question}

Chunks:
{chunk_list}

Return ONLY JSON, e.g.: [1, 3, 5, 8]"""


def _filter_relevant_chunks(chunks, question):
    """Use LLM to filter out irrelevant chunks before synthesis."""
    if len(chunks) <= 5:
        return chunks  # Not worth filtering small sets

    # Build compact chunk summaries for the relevance check
    chunk_list = []
    for i, chunk in enumerate(chunks, 1):
        preview = chunk.text[:150].replace("\n", " ")
        chunk_list.append(f"[{i}] {chunk.metadata.filename}: {preview}")
    chunk_text = "\n".join(chunk_list)

    try:
        response = generate(
            system_prompt="You filter document chunks by relevance. Return ONLY a JSON array of relevant chunk numbers.",
            user_prompt=RELEVANCE_PROMPT.format(question=question, chunk_list=chunk_text),
            temperature=0.0,
            max_tokens=1024,
        )
        relevant_ids = parse_json_response(response)
        if isinstance(relevant_ids, list) and relevant_ids:
            filtered = [chunks[i - 1] for i in relevant_ids if 1 <= i <= len(chunks)]
            logger.info(f"Relevance filter: {len(chunks)} → {len(filtered)} chunks")
            return filtered if filtered else chunks
    except Exception as e:
        logger.warning(f"Relevance filtering failed, using all chunks: {e}")

    return chunks


def synthesize_answer(state: AgentState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    sql_results = state.get("sql_results", [])
    question = state["question"]

    if not chunks and not sql_results:
        return {
            "answer": "I could not find any relevant information in the documents you have access to.",
            "citations": [],
        }

    # Filter out irrelevant chunks before synthesis
    chunks = _filter_relevant_chunks(chunks, question)

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[{i}] {chunk.metadata.filename}"
        if chunk.metadata.page is not None:
            source += f", page {chunk.metadata.page}"
        context_parts.append(f"{source}:\n{chunk.text}")
    if sql_results:
        context_parts.append(f"[Database query results]:\n{json.dumps(sql_results, indent=2)}")
    context = "\n\n".join(context_parts)

    answer = generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT_TEMPLATE.format(context=context, question=question),
        max_tokens=4096,
    )

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
    return {"answer": answer, "citations": citations}
