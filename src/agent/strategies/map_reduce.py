"""Map-Reduce strategy for exhaustive queries spanning many documents.

Map:    Extract relevant data from each document individually
Reduce: Combine all extractions into a final answer

This avoids the problem of dumping 100+ chunks into one LLM call
where details get lost in the noise.
"""
import asyncio
import logging

from src.agent.state import AgentState
from src.generation.llm_client import generate
from src.ingestion.embedder import embed_query
from src.retrieval.models import RetrievedChunk, ChunkMetadata
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

MAP_PROMPT = """Extract ONLY the information relevant to this question from the document below.
If the document contains no relevant information, respond with "NO_RELEVANT_DATA".
Be specific — include names, amounts, dates, locations, and contract numbers.

Question: {question}

Document ({filename}):
{content}"""

REDUCE_PROMPT = """Combine these per-document extractions into a complete, thorough answer.
Include ALL items found — do not summarize or omit any. Cite the source filename for each item.

Question: {question}

Extractions:
{extractions}"""


async def retrieve_map_reduce(
    state: AgentState,
    vector_store: VectorStore,
    top_k: int = 100,
) -> dict:
    """Map-reduce: extract from each doc individually, then combine."""
    question = state["question"]
    user_groups = state["user_groups"]
    doc_ids = state.get("allowed_doc_ids")

    # Step 1: Find relevant documents (same as sweep discovery)
    query_vector = await asyncio.to_thread(embed_query, question)

    # Check for date-specific query
    from src.agent.strategies.sweep import _extract_date_filter
    date_filter_docs = _extract_date_filter(question, vector_store)

    if date_filter_docs:
        relevant_doc_ids = date_filter_docs
        logger.info(f"Map-reduce: date filter matched {len(relevant_doc_ids)} documents")
    else:
        # Phase 1: Search summary embeddings for fast document discovery
        summary_results = vector_store.search(
            vector=query_vector, user_groups=user_groups,
            top_k=top_k, tier="summary", doc_ids=doc_ids,
        )
        # Fallback to xlarge if no summary embeddings exist yet
        if not summary_results:
            summary_results = vector_store.hybrid_search(
                vector=query_vector, text_query=question,
                user_groups=user_groups, top_k=top_k, tier="xlarge", doc_ids=doc_ids,
            )
            logger.info("Map-reduce: no summary embeddings found, falling back to xlarge search")

        candidate_doc_ids = list({c.metadata.doc_id for c in summary_results})
        doc_filenames = {c.metadata.doc_id: c.metadata.filename for c in summary_results}
        doc_scores = {c.metadata.doc_id: c.score for c in summary_results}
        logger.info(f"Map-reduce: {len(candidate_doc_ids)} candidate docs from summary/xlarge search")

        # Phase 2: Also search metadata for keyword matches to catch docs that
        # vector search missed (different terminology, lower embedding similarity)
        metadata_store = None
        try:
            from src.api.routes_ingest import get_metadata_store
            metadata_store = get_metadata_store()
        except Exception:
            pass

        if metadata_store:
            q_lower = question.lower()
            q_words = {w for w in q_lower.split() if len(w) > 3}

            # Search ALL documents by metadata to find ones vector search missed
            all_docs = await metadata_store.list_documents(user_groups)
            for doc in all_docs:
                if doc.doc_id in {d for d in candidate_doc_ids}:
                    continue  # already found by vector search
                meta = getattr(doc, 'metadata_tags', {}) or {}
                if not meta:
                    continue
                # Check if any metadata field matches query terms
                match = False
                for field in ["entities", "organizations", "topics", "identifiers"]:
                    for val in meta.get(field, []):
                        if val and (val.lower() in q_lower or any(w in val.lower() for w in q_words)):
                            match = True
                            break
                    if match:
                        break
                if match:
                    candidate_doc_ids.append(doc.doc_id)
                    doc_filenames[doc.doc_id] = doc.filename
                    doc_scores[doc.doc_id] = 0.1  # low score, found by metadata only

            # Score and rank all candidates
            scored_docs = []
            for did in candidate_doc_ids:
                doc_rec = await metadata_store.get_document(did)
                meta = getattr(doc_rec, 'metadata_tags', {}) or {} if doc_rec else {}
                meta_score = 0
                if meta:
                    for field, values in meta.items():
                        if field == "summary" or not isinstance(values, list):
                            continue
                        for val in values:
                            if val and val.lower() in q_lower:
                                meta_score += 2
                            elif val and any(w in val.lower() for w in q_words):
                                meta_score += 1
                combined = doc_scores.get(did, 0) + (meta_score * 0.1)
                scored_docs.append((did, combined))

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            relevant_doc_ids = [did for did, _ in scored_docs[:50]]
        else:
            relevant_doc_ids = candidate_doc_ids

        logger.info(f"Map-reduce: {len(relevant_doc_ids)} docs after metadata search (from {len(candidate_doc_ids)} candidates)")
        for did in relevant_doc_ids:
            logger.info(f"  - {doc_filenames.get(did, did)}")

    # Step 2: MAP — extract relevant data from each document in parallel
    async def map_document(doc_id: str) -> dict:
        chunks = await asyncio.to_thread(
            vector_store.get_chunks_by_doc, doc_id, 200, "large"
        )
        if not chunks:
            return {"doc_id": doc_id, "filename": "unknown", "extraction": ""}

        filename = chunks[0].metadata.filename
        content = "\n\n".join(c.text for c in chunks)

        # Truncate if too long for a single LLM call (256K context target)
        max_content = 200000
        if len(content) > max_content:
            content = content[:max_content] + "\n... [truncated]"

        try:
            extraction = await asyncio.to_thread(
                generate,
                system_prompt="You extract specific data from documents. Be thorough and precise.",
                user_prompt=MAP_PROMPT.format(
                    question=question,
                    filename=filename,
                    content=content,
                ),
                temperature=0.0,
                max_tokens=8192,
            )

            if "NO_RELEVANT_DATA" in extraction:
                return {"doc_id": doc_id, "filename": filename, "extraction": ""}

            return {"doc_id": doc_id, "filename": filename, "extraction": extraction.strip()}
        except Exception as e:
            logger.warning(f"Map failed for {filename}: {e}")
            return {"doc_id": doc_id, "filename": filename, "extraction": ""}

    # Run map in parallel (bounded concurrency)
    from src.config import settings as _settings
    semaphore = asyncio.Semaphore(_settings.llm_concurrency)

    async def bounded_map(doc_id):
        async with semaphore:
            return await map_document(doc_id)

    map_results = await asyncio.gather(*[bounded_map(did) for did in relevant_doc_ids])

    # Filter out empty extractions
    valid_extractions = [r for r in map_results if r["extraction"]]
    logger.info(f"Map-reduce: {len(valid_extractions)}/{len(map_results)} documents had relevant data")
    for r in map_results:
        status = "RELEVANT" if r["extraction"] else "NO_DATA"
        logger.info(f"  [{status}] {r['filename']}")

    if not valid_extractions:
        return {
            "retrieved_chunks": [],
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    # Step 3: REDUCE — build extractions into a synthetic chunk for the synthesizer
    extraction_text = "\n\n".join(
        f"[{r['filename']}]:\n{r['extraction']}"
        for r in valid_extractions
    )

    # Create a synthetic chunk containing the combined map results
    reduce_chunk = RetrievedChunk(
        text=f"Map-Reduce Results ({len(valid_extractions)} documents processed):\n\n{extraction_text}",
        score=1.0,
        metadata=ChunkMetadata(
            doc_id="map-reduce",
            filename="map_reduce_synthesis",
            doc_type="synthesis",
            chunk_index=0,
            start_char=0,
            acl_groups=["ALL"],
        ),
    )

    return {
        "retrieved_chunks": [reduce_chunk],
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
