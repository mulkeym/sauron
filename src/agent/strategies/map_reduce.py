"""Map-Reduce strategy for exhaustive queries spanning many documents.

Map:    Extract relevant data from each document individually
Reduce: Combine all extractions into a final answer

This avoids the problem of dumping 100+ chunks into one LLM call
where details get lost in the noise.
"""
import asyncio
import logging
import math

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


# Query words too common to discriminate documents — interrogatives, generic
# domain vocabulary. Calendar months are handled by IDF (they end up in nearly
# every doc's metadata), so they don't need to be listed here.
_STOPWORDS = {
    "what", "which", "when", "where", "who", "whom", "whose", "did", "does",
    "the", "and", "for", "all", "any", "list", "show", "give", "find", "about",
    "contract", "contracts", "award", "awarded", "awards",
}


def _metadata_blob(metadata: dict) -> str:
    """Flatten a metadata_tags dict into one lowercase searchable string."""
    if not metadata:
        return ""
    parts = []
    for key, val in metadata.items():
        if key == "summary":
            continue
        if isinstance(val, list):
            parts.extend(str(v) for v in val)
        elif isinstance(val, str):
            parts.append(val)
    return " ".join(parts).lower()


def _query_terms(question: str) -> set[str]:
    """Significant question terms (length > 3, not stopwords)."""
    terms = {w.strip(".,?!:;\"'()").lower() for w in question.split()}
    return {t for t in terms if len(t) > 3 and t not in _STOPWORDS}


def _term_idf(query_terms, docs_metadata) -> dict:
    """IDF weight per term from how many candidate docs' metadata contain it.

    Terms in nearly every doc (e.g. 'march' against date-named files) collapse
    toward 0; rare, specific terms (entity names, identifiers) get high weight.
    """
    n = len(docs_metadata)
    if n == 0:
        return {t: 1.0 for t in query_terms}
    blobs = [_metadata_blob(m) for m in docs_metadata]
    df = {t: 0 for t in query_terms}
    for blob in blobs:
        for t in query_terms:
            if t and t in blob:
                df[t] += 1
    return {t: math.log((n + 1) / (df[t] + 1)) for t in query_terms}


def _meta_match_score(query_terms, metadata, term_idf) -> float:
    """Sum of IDF weights for query terms present in this doc's metadata."""
    blob = _metadata_blob(metadata)
    return sum(term_idf.get(t, 0.0) for t in query_terms if t and t in blob)


def _rrf_fuse(rankings, k: int = 60) -> dict:
    """Reciprocal rank fusion: merge ranked id-lists into a scale-invariant
    score, so a strong vector match and a strong metadata match are comparable
    regardless of their raw value ranges."""
    scores: dict = {}
    for ranking in rankings:
        for rank, did in enumerate(ranking):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _normalize_relevance(scores: dict) -> dict:
    """Scale a score dict to 0..1 by its max, so the most relevant doc reads
    ~1.0 and the rest scale proportionally. Used to give the playground and
    citations a meaningful relevance instead of the 0.0 that fetched-by-id
    chunks carry."""
    if not scores:
        return {}
    top = max(scores.values())
    if top <= 0:
        return {d: 0.0 for d in scores}
    return {d: s / top for d, s in scores.items()}


async def _prefilter_by_summary(summaries, question, judge, concurrency: int) -> list:
    """Keep doc_ids whose summary a cheap judge deems relevant.

    Fails open: if the judge call errors, the doc is kept rather than silently
    dropped — the full MAP pass remains the source of truth.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    doc_ids = list(summaries.keys())

    async def check(doc_id):
        async with sem:
            try:
                return doc_id, await judge(question, summaries[doc_id])
            except Exception:
                return doc_id, True  # fail open
    results = await asyncio.gather(*[check(d) for d in doc_ids])
    return [d for d, keep in results if keep]


async def _map_documents(doc_ids, map_one, concurrency: int, retry_concurrency: int, max_retries: int = 1):
    """Run ``map_one`` over ``doc_ids``, re-attempting failures at reduced
    concurrency before returning.

    ``map_one(doc_id)`` must return a dict with a ``status`` of ``"ok"``,
    ``"empty"`` (read fine, no relevant content), or ``"failed"`` (timeout/
    error). Returns ``(results, still_failed_doc_ids)`` — failures that survive
    every retry are reported, never silently counted as "no data".
    """
    async def run(ids, conc):
        sem = asyncio.Semaphore(max(1, conc))

        async def bounded(did):
            async with sem:
                return await map_one(did)
        return await asyncio.gather(*[bounded(d) for d in ids])

    results = await run(doc_ids, concurrency)
    by_id = {r["doc_id"]: r for r in results}

    failed = [r["doc_id"] for r in results if r["status"] == "failed"]
    retries = 0
    while failed and retries < max_retries:
        retries += 1
        logger.info(f"Map-reduce: retrying {len(failed)} failed docs (attempt {retries}) at concurrency {retry_concurrency}")
        retry_results = await run(failed, retry_concurrency)
        for r in retry_results:
            by_id[r["doc_id"]] = r
        failed = [r["doc_id"] for r in retry_results if r["status"] == "failed"]

    return list(by_id.values()), failed


async def retrieve_map_reduce(
    state: AgentState,
    vector_store: VectorStore,
    top_k: int = 500,
) -> dict:
    """Map-reduce: extract from each doc individually, then combine."""
    question = state["question"]
    user_groups = state["user_groups"]
    doc_ids = state.get("allowed_doc_ids")

    # Step 1: Find relevant documents (same as sweep discovery)
    query_vector = await asyncio.to_thread(embed_query, question)

    # Check for date-specific query
    from src.agent.strategies.sweep import _extract_date_filter
    date_filter_docs = _extract_date_filter(question, vector_store, user_groups)

    if date_filter_docs:
        logger.info(f"Map-reduce: date filter found {len(date_filter_docs)} docs mentioning the date")

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

    # PRF: expand query with terms from top results, search again to find more docs
    try:
        from src.retrieval.prf import expand_query_with_prf
        expanded_query, expanded_vector = await expand_query_with_prf(
            question, query_vector, user_groups, vector_store, doc_ids,
        )
        if expanded_query != question:
            prf_results = vector_store.search(
                vector=expanded_vector, user_groups=user_groups,
                top_k=top_k, tier="summary", doc_ids=doc_ids,
            )
            if not prf_results:
                prf_results = vector_store.hybrid_search(
                    vector=expanded_vector, text_query=expanded_query,
                    user_groups=user_groups, top_k=top_k, tier="xlarge", doc_ids=doc_ids,
                )
            prf_new = 0
            for c in prf_results:
                if c.metadata.doc_id not in doc_scores:
                    candidate_doc_ids.append(c.metadata.doc_id)
                    doc_filenames[c.metadata.doc_id] = c.metadata.filename
                    doc_scores[c.metadata.doc_id] = c.score
                    prf_new += 1
            if prf_new:
                logger.info(f"PRF added {prf_new} new candidate docs")
    except Exception as e:
        logger.debug(f"PRF skipped: {e}")

    # Fetch feedback boosts from past similar queries
    feedback_boosts = {}
    try:
        from src.retrieval.feedback import get_feedback_boosts
        feedback_boosts = await get_feedback_boosts(query_vector, user_groups)
    except Exception:
        pass

    # Phase 2: Also search metadata for keyword matches to catch docs that
    # vector search missed (different terminology, lower embedding similarity)
    metadata_store = None
    try:
        from src.api.routes_ingest import get_metadata_store
        metadata_store = get_metadata_store()
    except Exception:
        pass

    meta_by_id: dict = {}
    doc_relevance: dict = {}
    if metadata_store:
        q_terms = _query_terms(question)
        all_docs = await metadata_store.list_documents(user_groups)
        allowed_set = set(doc_ids) if doc_ids else None

        # Collect metadata for every candidate, and discover docs that vector
        # search missed but whose metadata contains a significant query term.
        existing_candidates = set(candidate_doc_ids)
        for doc in all_docs:
            if allowed_set and doc.doc_id not in allowed_set:
                continue  # not in selected dataset
            meta = getattr(doc, 'metadata_tags', {}) or {}
            if doc.doc_id not in existing_candidates:
                blob = _metadata_blob(meta)
                if not any(t in blob for t in q_terms):
                    continue  # not a vector hit and no significant term match
                candidate_doc_ids.append(doc.doc_id)
                existing_candidates.add(doc.doc_id)
                doc_filenames[doc.doc_id] = doc.filename
            meta_by_id[doc.doc_id] = meta
            doc_filenames.setdefault(doc.doc_id, doc.filename)

        # Weight query terms by rarity across the candidate set, so calendar /
        # common words (e.g. "march" against date-named files) collapse to ~0
        # while specific terms (entity names, identifiers) carry the signal.
        candidate_meta = [meta_by_id.get(did, {}) for did in candidate_doc_ids]
        term_idf = _term_idf(q_terms, candidate_meta)

        # Rank candidates two ways — vector similarity and IDF-weighted metadata
        # match — then fuse the rankings (reciprocal rank fusion) so the two
        # incomparable score scales no longer fight. A real vector hit is no
        # longer swamped by a flat metadata grant, and a metadata-only "march"
        # match scores 0 instead of the old 0.1 floor that admitted everything.
        vector_ranked = sorted(
            [d for d in candidate_doc_ids if d in doc_scores],
            key=lambda d: doc_scores[d], reverse=True,
        )
        meta_match = {
            d: _meta_match_score(q_terms, meta_by_id.get(d, {}), term_idf)
            for d in candidate_doc_ids
        }
        meta_ranked = sorted(
            [d for d in candidate_doc_ids if meta_match[d] > 0],
            key=lambda d: meta_match[d], reverse=True,
        )
        fused = _rrf_fuse([vector_ranked, meta_ranked])
        for did in candidate_doc_ids:
            fused[did] = fused.get(did, 0.0) + feedback_boosts.get(did, 0.0)

        ranked = sorted(candidate_doc_ids, key=lambda d: fused.get(d, 0.0), reverse=True)

        # Relative cutoff (no absolute floor): keep docs scoring within a
        # fraction of the top fused score; drop negative-feedback docs.
        relevant_doc_ids = []
        skipped_by_feedback = 0
        if ranked:
            top = fused.get(ranked[0], 0.0)
            threshold = top * 0.3 if top > 0 else 0.0
            for did in ranked:
                if feedback_boosts.get(did, 0) < 0:
                    skipped_by_feedback += 1
                    continue
                score = fused.get(did, 0.0)
                if score > 0 and score >= threshold:
                    relevant_doc_ids.append(did)
        if skipped_by_feedback:
            logger.info(f"Map-reduce: feedback excluded {skipped_by_feedback} previously-irrelevant docs")
        # Normalized fused score per candidate, for display/citations.
        doc_relevance = _normalize_relevance({d: fused.get(d, 0.0) for d in candidate_doc_ids})
    else:
        if feedback_boosts:
            scored = [(did, doc_scores.get(did, 0) + feedback_boosts.get(did, 0)) for did in candidate_doc_ids]
            scored.sort(key=lambda x: x[1], reverse=True)
            relevant_doc_ids = [did for did, _ in scored]
        else:
            relevant_doc_ids = candidate_doc_ids
        doc_relevance = _normalize_relevance({
            d: doc_scores.get(d, 0.0) + feedback_boosts.get(d, 0.0) for d in candidate_doc_ids
        })

    # Merge date-matched docs into the list
    if date_filter_docs:
        existing = set(relevant_doc_ids)
        for did in date_filter_docs:
            if did not in existing:
                relevant_doc_ids.append(did)
        logger.info(f"Map-reduce: {len(relevant_doc_ids)} docs after merging date filter")
    else:
        logger.info(f"Map-reduce: {len(relevant_doc_ids)} docs pass relevance threshold (from {len(candidate_doc_ids)} candidates)")
    for did in relevant_doc_ids:
        logger.info(f"  - {doc_filenames.get(did, did)}")

    from src.config import settings as _settings

    # Pre-MAP relevance gate: a cheap YES/NO judgment on each doc's summary so
    # we don't spend a full extraction call on clearly-irrelevant docs. Docs
    # without a summary, and date-matched docs, bypass the gate. The gate fails
    # open, so a timed-out judgment keeps the doc for the full MAP pass.
    summaries = {}
    for did in relevant_doc_ids:
        summ = (meta_by_id.get(did) or {}).get("summary", "")
        if isinstance(summ, str) and summ.strip():
            summaries[did] = summ.strip()
    gate_exempt = set(date_filter_docs or []) | (set(relevant_doc_ids) - set(summaries))
    if summaries:
        async def _judge(q: str, summary: str) -> bool:
            verdict = await asyncio.to_thread(
                generate,
                system_prompt="You decide whether a document could help answer a question. Reply with only YES or NO.",
                user_prompt=(
                    f"Question: {q}\n\nDocument summary:\n{summary}\n\n"
                    "Could this document contain information relevant to the question? Answer YES or NO."
                ),
                temperature=0.0,
                max_tokens=4,
            )
            return verdict.strip().upper().startswith("Y")

        gate_concurrency = max(_settings.llm_concurrency * 2, 8)
        kept = set(await _prefilter_by_summary(summaries, question, _judge, concurrency=gate_concurrency))
        before = len(relevant_doc_ids)
        relevant_doc_ids = [d for d in relevant_doc_ids if d in kept or d in gate_exempt]
        logger.info(f"Map-reduce: pre-MAP relevance gate kept {len(relevant_doc_ids)}/{before} docs")

    # Step 2: MAP — extract relevant data from each document in parallel
    async def map_document(doc_id: str) -> dict:
        chunks = await asyncio.to_thread(
            vector_store.get_chunks_by_doc, doc_id, 200, "large"
        )
        if not chunks:
            return {"doc_id": doc_id, "filename": doc_filenames.get(doc_id, "unknown"), "extraction": "", "status": "empty"}

        filename = chunks[0].metadata.filename
        content = "\n\n".join(c.text for c in chunks)

        # Truncate if too long for a single LLM call
        max_content = _settings.llm_max_context
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
        except Exception as e:
            # A timeout/error is NOT "no data" — flag it so it can be retried
            # and, if it still fails, reported instead of silently dropped.
            logger.warning(f"Map failed for {filename}: {e}")
            return {"doc_id": doc_id, "filename": filename, "extraction": "", "status": "failed"}

        if "NO_RELEVANT_DATA" in extraction:
            return {"doc_id": doc_id, "filename": filename, "extraction": "", "status": "empty"}
        return {"doc_id": doc_id, "filename": filename, "extraction": extraction.strip(), "status": "ok"}

    # Run MAP with a retry queue: failures are re-attempted at reduced
    # concurrency before reduce, so transient overload doesn't lose real data.
    map_results, still_failed = await _map_documents(
        relevant_doc_ids,
        map_document,
        concurrency=_settings.llm_concurrency,
        retry_concurrency=max(1, _settings.llm_concurrency // 2),
        max_retries=1,
    )

    valid_extractions = [r for r in map_results if r["status"] == "ok"]
    logger.info(f"Map-reduce: {len(valid_extractions)}/{len(map_results)} documents had relevant data")
    for r in map_results:
        logger.info(f"  [{r['status'].upper()}] {r['filename']}")

    # Build a user-visible notice for docs that could not be analyzed even
    # after retry, so an incomplete answer never looks like a complete one.
    incomplete_note = ""
    if still_failed:
        failed_names = [doc_filenames.get(d, d) for d in still_failed]
        logger.warning(f"Map-reduce: {len(still_failed)} docs could not be analyzed after retry: {failed_names}")
        shown = ", ".join(failed_names[:10]) + ("..." if len(failed_names) > 10 else "")
        incomplete_note = (
            f"⚠️ NOTE: {len(still_failed)} document(s) could not be analyzed "
            f"(LLM timeouts/errors) and are NOT reflected in this answer; results may be "
            f"incomplete: {shown}\n\n"
        )

    if not valid_extractions:
        chunks = []
        if incomplete_note:
            chunks = [RetrievedChunk(
                text=incomplete_note.strip(), score=1.0,
                metadata=ChunkMetadata(
                    doc_id="map-reduce", filename="map_reduce_synthesis",
                    doc_type="synthesis", chunk_index=0, start_char=0, acl_groups=["ALL"],
                ),
            )]
        return {
            "retrieved_chunks": chunks,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
            "warnings": [incomplete_note.strip()] if incomplete_note else [],
            "doc_relevance": doc_relevance,
        }

    # Step 3: REDUCE — hierarchical reduce if extractions are too large
    from src.config import settings as _cfg
    MAX_REDUCE_CHARS = int(_cfg.llm_max_context * 0.6)  # 60% of context for reduce batches

    extraction_text = "\n\n".join(
        f"[{r['filename']}]:\n{r['extraction']}"
        for r in valid_extractions
    )

    if len(extraction_text) > MAX_REDUCE_CHARS:
        # Hierarchical reduce: batch extractions, reduce each batch, then combine
        logger.info(f"Map-reduce: extraction too large ({len(extraction_text):,} chars), running hierarchical reduce")

        # Split into batches that fit in the LLM context
        batches = []
        current_batch = []
        current_size = 0
        for r in valid_extractions:
            entry = f"[{r['filename']}]:\n{r['extraction']}"
            if current_size + len(entry) > MAX_REDUCE_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            current_batch.append(entry)
            current_size += len(entry)
        if current_batch:
            batches.append(current_batch)

        logger.info(f"Map-reduce: reducing {len(valid_extractions)} extractions in {len(batches)} batches")

        # Reduce each batch in parallel
        async def reduce_batch(batch_entries, batch_num):
            batch_text = "\n\n".join(batch_entries)
            try:
                summary = await asyncio.to_thread(
                    generate,
                    system_prompt="You combine document extractions into a comprehensive summary. Include ALL items — do not omit any. Cite source filenames.",
                    user_prompt=REDUCE_PROMPT.format(question=question, extractions=batch_text),
                    temperature=0.0,
                    max_tokens=8192,
                )
                return summary.strip()
            except Exception as e:
                logger.warning(f"Reduce batch {batch_num} failed: {e}")
                return batch_text[:MAX_REDUCE_CHARS]  # fallback: return raw

        batch_results = await asyncio.gather(
            *[reduce_batch(b, i) for i, b in enumerate(batches)]
        )
        extraction_text = "\n\n".join(batch_results)
        logger.info(f"Map-reduce: hierarchical reduce complete, {len(extraction_text):,} chars")

    # Create a synthetic chunk containing the combined map results. Any
    # incompleteness notice is prepended so synthesis carries it into the answer.
    reduce_chunk = RetrievedChunk(
        text=f"{incomplete_note}Map-Reduce Results ({len(valid_extractions)} documents processed):\n\n{extraction_text}",
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
        "warnings": [incomplete_note.strip()] if incomplete_note else [],
        "doc_relevance": doc_relevance,
    }
