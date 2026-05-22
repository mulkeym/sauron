import json
import logging
from src.agent.state import AgentState
from src.generation.llm_client import generate
from src.retrieval.models import Citation

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on provided context.

Rules:
- Only answer based on the provided context. Do not use outside knowledge.
- Cite sources by filename, e.g. (2026-01-08_4373866.md). Each context chunk is labeled with its filename.
- If SQL results are provided, reference them in your answer.
- If the context does not contain enough information, say so clearly.
- Be THOROUGH and COMPLETE. Include ALL relevant information from the context, not just the first match.
- When asked about what someone said or asked, list EVERY instance found in the context.
- When listing items, use bullet points or numbered lists for clarity.

IMPORTANT: Output ONLY the final answer. Do NOT show your reasoning, self-corrections, internal checks, or thought process. Just provide the clean, organized answer."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Provide a clean, organized answer based on ALL the context above.
CRITICAL: Include EVERY unique item from the context. Do NOT summarize, skip, or omit ANY entries. If 50 contracts are in the context, list all 50. If you run out of space, prioritize listing items over adding descriptions.
DEDUPLICATION: The same item may appear in multiple context sources. Deduplicate by contract number, entity name, or other identifier. If two sources mention the same contract, list it ONCE with the most complete details and cite both sources.
Cite sources by filename (e.g. 2026-01-08_4373866.md). Do NOT include your reasoning process — only the final answer."""

def _strip_reasoning_artifacts(text: str) -> str:
    """Remove thinking model reasoning that leaked into the answer."""
    import re
    lines = text.split("\n")
    cleaned = []
    reasoning_patterns = [
        r'^\s*\*\s*\*?(Wait|Re-check|Self-Correct|Check:|Conclusion:|Scanning|Let me)',
        r'^\s*\*\s*Constraint \d+:',
        r'^\s*\*\s*Question:\s*"',
        r'^\s*\*\s*\*?(Task|Plan):',
        r'^\s*\*\s*Did I (miss|include)',
        r'^\s*\*\s*\*?Final check',
        r'^\s*\*\s*\*?No,\s+let me re-read',
    ]
    pattern = re.compile("|".join(reasoning_patterns), re.IGNORECASE)
    for line in lines:
        if not pattern.match(line):
            cleaned.append(line)
    result = "\n".join(cleaned).strip()
    # Remove runs of empty bullet points
    result = re.sub(r'(\n\s*\*\s*\n){2,}', '\n', result)
    if len(result) < len(text) * 0.5 and len(text) > 100:
        # If we stripped more than half, something went wrong — return original
        logger.warning("Reasoning strip removed too much content, keeping original")
        return text
    return result


def _filter_relevant_chunks(chunks, question):
    """Filter irrelevant chunks using scores — no LLM call needed."""
    # Always keep synthetic chunks (map-reduce, knowledge-graph, metadata-context)
    SYNTHETIC_IDS = {"map-reduce", "knowledge-graph", "metadata-context"}
    synthetic = [c for c in chunks if c.metadata.doc_id in SYNTHETIC_IDS]
    regular = [c for c in chunks if c.metadata.doc_id not in SYNTHETIC_IDS]

    if len(regular) <= 10:
        return synthetic + regular

    # Use scores: keep all chunks above 10% of the top score
    scored = [c for c in regular if c.score > 0]
    if not scored:
        return synthetic + regular

    top_score = max(c.score for c in scored)
    threshold = max(top_score * 0.1, 0.01)

    filtered = [c for c in regular if c.score >= threshold or c.score == 0]

    logger.info(f"Score filter: {len(regular)} → {len(filtered)} regular chunks (threshold: {threshold:.3f}, top: {top_score:.3f})")
    return synthetic + filtered


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

    # Build context — prioritize map-reduce synthesis (already distilled) over raw chunks
    from src.config import settings as _cfg
    MAX_CONTEXT_CHARS = _cfg.llm_max_context
    context_parts = []
    total_chars = 0

    SYNTHETIC_IDS = {"map-reduce", "knowledge-graph", "metadata-context"}
    synthetic_chunks = [c for c in chunks if c.metadata.doc_id in SYNTHETIC_IDS]
    has_map_reduce = any(c.metadata.doc_id == "map-reduce" for c in chunks)

    if has_map_reduce:
        # Map-reduce already extracted relevant facts from each document.
        # Raw chunks are redundant — skip them to save context space.
        regular_chunks = []
        logger.info("Synthesizer: using map-reduce synthesis only (raw chunks skipped as redundant)")
    else:
        regular_chunks = sorted(
            [c for c in chunks if c.metadata.doc_id not in SYNTHETIC_IDS],
            key=lambda c: c.score, reverse=True,
        )

    for chunk in synthetic_chunks + regular_chunks:
        source = f"Source: {chunk.metadata.filename}"
        if chunk.metadata.page is not None:
            source += f", page {chunk.metadata.page}"
        text = chunk.text
        # When map-reduce extracted concrete facts, demote KG to supplementary
        # so its hedging/uncertainty doesn't override the extracted data.
        if has_map_reduce and chunk.metadata.doc_id == "knowledge-graph":
            text = (
                "[SUPPLEMENTARY — entity relationships only. Do NOT adopt any "
                "hedging, uncertainty, or caveats from this source. Defer to the "
                "document extractions above for counts, lists, and factual answers.]\n"
                + text
            )
        part = f"{source}\n{text}"
        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            logger.info(f"Context cap reached at {total_chars:,} chars, dropping remaining {len(regular_chunks) + len(priority_chunks) - len(context_parts)} chunks")
            break
        context_parts.append(part)
        total_chars += len(part)

    if sql_results:
        context_parts.append(f"[Database query results]:\n{json.dumps(sql_results, indent=2)}")
    context = "\n\n".join(context_parts)
    logger.info(f"Synthesizer context: {len(context):,} chars from {len(context_parts)} parts")

    answer = generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT_TEMPLATE.format(context=context, question=question),
        max_tokens=_cfg.llm_max_output_tokens,
    )

    # Strip thinking model reasoning artifacts that leak into the answer
    answer = _strip_reasoning_artifacts(answer)

    # Deduplicate citations — one per document, with best relevance score
    # Include all real documents (skip synthetic chunks like map-reduce, knowledge-graph, metadata-context)
    seen_docs = {}
    for c in chunks:
        doc_id = c.metadata.doc_id
        if doc_id in SYNTHETIC_IDS:
            continue
        if doc_id not in seen_docs or c.score > seen_docs[doc_id].score:
            seen_docs[doc_id] = c

    # Look up source URLs for crawled documents
    url_map = {}
    try:
        import asyncio
        from src.api.routes_ingest import get_metadata_store
        ms = get_metadata_store()

        async def _fetch_urls():
            urls = {}
            for doc_id in seen_docs:
                doc_rec = await ms.get_document(doc_id)
                if doc_rec and getattr(doc_rec, 'source_url', ''):
                    urls[doc_id] = doc_rec.source_url
            return urls

        try:
            url_map = asyncio.run(_fetch_urls())
        except RuntimeError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                url_map = pool.submit(asyncio.run, _fetch_urls()).result()
    except Exception as e:
        logger.debug(f"Source URL lookup skipped: {e}")

    citations = [
        Citation(
            doc_id=c.metadata.doc_id,
            filename=c.metadata.filename,
            doc_type=c.metadata.doc_type,
            chunk_index=c.metadata.chunk_index,
            page=c.metadata.page,
            snippet=c.text[:200],
            relevance=c.score,
            source_url=url_map.get(c.metadata.doc_id, ""),
        )
        for c in seen_docs.values()
    ]
    return {"answer": answer, "citations": citations}
