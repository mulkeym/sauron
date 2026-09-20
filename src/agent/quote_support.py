"""Conservative quotation checks and bounded retrieval of original support.

This checks literal attribution, not semantic entailment of paraphrases.
"""
from __future__ import annotations

import re
import unicodedata


def normalize_quote(text: str) -> str:
    # PDF extraction may omit spaces or wrap lines. Preserve case, punctuation,
    # negation, numbers and direction; never fuzzy-match a different statement.
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).replace("\u00ad", "")
    # Do not merge separate numeric values ("9 0" must not become "90").
    return re.sub(r"(?<!\d) | (?!\d)", "", text)


def primary_text(text: str) -> str:
    # Ingestion prepends generated document summaries to original text chunks.
    # Those summaries cannot establish an exact quotation from the original.
    if text.startswith("Document:") and "\nSummary:" in text.split("\n\n", 1)[0]:
        return text.partition("\n\n")[2]
    return text


def literal_spans(text):
    # Check inline code as literal source content too: invented commands can be
    # dangerous even when the surrounding prose has an otherwise valid citation.
    for match in re.finditer(r'`([^`\n]+)`|["“]([^"”]+)["”]', text):
        yield match[1] or match[2], bool(match[1])


def unsupported_quotes(answer, citations, question=""):
    # Mermaid strings are generated drawing syntax, not asserted verbatim quotes.
    # Other code (especially commands) remains subject to literal validation.
    answer = re.sub(r"(?m)^\s*(`{3,}|~{3,})mermaid[^\n]*\n[\s\S]*?^\s*\1\s*$", "", answer)
    by_id = {c.evidence_id: c for c in citations}
    normalized = {eid: normalize_quote(primary_text(c.snippet)) for eid, c in by_id.items()
                  if c.source_kind in {"document", "query_result"}}
    failures = []
    for paragraph in re.split(r"\n\s*\n", answer):
        ids = set(re.findall(r"\[(E[A-Za-z0-9_-]+)\]", paragraph))
        for quote, is_code in literal_spans(paragraph):
            needle = normalize_quote(quote)
            # Quoting a term from the user's question is not a source quotation.
            if not needle or (not is_code and needle.rstrip(".,;:!?") in normalize_quote(question)):
                continue
            # Filenames are cited metadata, often formatted as inline code.
            if any(quote == by_id[eid].filename for eid in ids if eid in by_id):
                continue
            if not any(needle in normalized.get(eid, "") for eid in ids):
                failures.append({"quote": quote, "evidence_ids": sorted(ids)})
    return failures


def has_source_quote(answer, question="", citations=None):
    # Inline code is a literal quotation too, provided it matches an original
    # passage cited in its own paragraph. A filename metadata match alone does
    # not qualify. Fenced generated code (including Mermaid) never counts.
    answer = re.sub(r"(?m)^\s*(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\s*\1\s*$", "", answer)
    by_id = {c.evidence_id: c for c in citations or [] if c.source_kind in {'document', 'query_result'}}
    for paragraph in re.split(r"\n\s*\n", answer):
        refs = re.findall(r'\[(E[A-Za-z0-9_-]+)\]', paragraph)
        for text, is_code in literal_spans(paragraph):
            needle = normalize_quote(text)
            if not needle or (not is_code and needle in normalize_quote(question)):
                continue
            if not is_code:
                return True  # Attribution is checked separately by unsupported_quotes.
            if any(ref in by_id and text != by_id[ref].filename
                   and needle in normalize_quote(primary_text(by_id[ref].snippet)) for ref in refs):
                return True
    return False


def needs_quoted_support(state):
    # General troubleshooting can be a cited paraphrase. Literal quotations and
    # commands are still checked independently, whether or not one is required.
    return (state.get("technical_intent") in {"procedure", "command_reference"}
            or (state.get("technical_intent") == "troubleshooting"
                and bool(re.search(r"\b(?:commands?|syntax|procedure|step[- ]by[- ]step)\b", state.get("question", ""), re.I)))
            or bool(re.search(r"\b(?:directions?|upstream|downstream|arrows?)\b|\btwo\s+lines\b",
                              state.get("question", ""), re.I)))


def retrieve_quote_support(state, pack, failures):
    """Search at most two already cited, authorized documents (256 rows each).

    Exact quote matches plus up to six reranked primary spans are new evidence
    for regeneration. Never substitute a citation into the old answer.
    """
    import asyncio
    from src.api.routes_ingest import get_metadata_store, get_vector_store
    from src.agent.state import chunk_key

    groups = state.get("user_groups", [])
    allowed = set(state.get("allowed_doc_ids") or [])
    if not groups or not allowed:
        return None
    refs = {eid for f in failures for eid in f["evidence_ids"]}
    doc_ids = list(dict.fromkeys(c.doc_id for c in pack.citations
                                if c.evidence_id in refs and c.doc_id in allowed))[:2]
    store = get_metadata_store()

    async def authorized():
        result = []
        for doc_id in doc_ids:
            doc = await store.get_document(doc_id)
            revision = state.get("edition_decisions", {}).get(doc_id, {}).get("source_revision")
            if (doc is not None and revision and doc.content_hash == revision
                    and ("ALL" in groups or set(doc.acl_groups).intersection(groups))
                    and (not state.get("dataset_id") or doc.dataset_id == state["dataset_id"])):
                result.append(doc_id)
        return result

    doc_ids = asyncio.run(authorized())
    vs = get_vector_store()
    candidates = []
    for doc_id in doc_ids:
        chunks, _ = vs.read_document_page(doc_id, groups, limit=256)
        candidates.extend(c for c in chunks if c.metadata.content_type == "text"
                          and c.metadata.chunk_size_tier == "medium")
    if not candidates:
        return None
    needles = [normalize_quote(f["quote"]) for f in failures[:8] if f["quote"]]
    exact = [c for c in candidates if any(n in normalize_quote(primary_text(c.text)) for n in needles)]
    if needles and not exact:
        return None
    # Reuse the existing reranker to recover nearby subject matter as well as
    # the quoted sentence (e.g. upstream/downstream versus a flow-path table).
    vs.rerank_chunks(candidates, state["question"], top_n=len(candidates))
    selected = exact[:8] + sorted(candidates, key=lambda c: -c.score)[:6]
    existing = {chunk_key(c): c for c in state.get("retrieved_chunks", [])}
    priority = max((c.score for c in existing.values()), default=0) + 1
    for c in selected:
        existing[chunk_key(c)] = c.model_copy(update={"score": priority})
    return {**state, "retrieved_chunks": list(existing.values())}


def repair_quotation_punctuation(answer, citations):
    """Move a stylistic trailing comma outside an otherwise exact quotation.

    No fuzzy matching: words, case, numbers, commands and internal punctuation
    are unchanged. Only a non-code quote ending in a letter plus comma qualifies.
    """
    from src.citations import map_prose
    originals = {c.evidence_id: normalize_quote(primary_text(c.snippet)) for c in citations
                 if c.source_kind in {'document', 'query_result'}}
    def paragraph(text):
        refs = re.findall(r'\[(E[A-Za-z0-9_-]+)\]', text)
        def replace(match):
            quoted = match[2]
            exact = normalize_quote(quoted)
            base = normalize_quote(quoted[:-1])
            if (len(quoted) > 1 and quoted[-2].isalpha()
                    and not any(exact in originals.get(ref, '') for ref in refs)
                    and any(base in originals.get(ref, '') for ref in refs)):
                return match[1] + quoted[:-1] + match[3] + ','
            return match[0]
        return map_prose(text, lambda prose: re.sub(r'(["“])([^"”\n]+,)(["”])', replace, prose))
    # Retain paragraph boundaries; quotations must match their own citation.
    return map_prose(answer, lambda prose: ''.join(part if i % 2 else paragraph(part)
                   for i, part in enumerate(re.split(r'(\n\s*\n)', prose))))
