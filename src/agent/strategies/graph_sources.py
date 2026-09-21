"""Use authorized graph names as search hints, never as procedural evidence."""
from __future__ import annotations
import asyncio
import re
from src.agent.state import chunk_key
from src.agent.profiles import profile_for_state
from src.config import settings


def _tokens(text):
    return set(re.findall(r'[a-z0-9]+', text.casefold())) - {
        'i', 'a', 'an', 'the', 'was', 'told', 'to', 'how', 'do', 'it', 'can', 'you',
        'please', 'what', 'is', 'are', 'my', 'with', 'for', 'of', 'in', 'and', 'me'}


def select_hints(question, hints):
    """The graph already ranked the matches; prefer shared subject words."""
    candidates, seen = [], set()
    words = _tokens(question)
    for hint in hints[:40]:
        term = hint.get('term', '')
        files = hint.get('filenames', [])
        if not isinstance(term, str) or not 2 <= len(term) <= 120 or '\n' in term or not files:
            continue
        key = ''.join(sorted(_tokens(term)))
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append((len(words & _tokens(term)), {'term': term, 'filenames': files}))
    candidates.sort(key=lambda item: -item[0])
    if not candidates:
        return []
    best = candidates[0][0]
    return [h for score, h in candidates if score == best][:3 if best else 2]


async def retrieve_graph_sources(state, vector_store, metadata_store):
    profile = profile_for_state(state)
    allowed = set(state.get('allowed_doc_ids') or [])
    groups = state.get('user_groups', [])
    if (not allowed or not groups or state.get('skip_graph')
            or (profile and not profile.graph_enrichment)
            or str(state.get('query_type', '')) not in {'lookup', 'procedure', 'troubleshooting'}):
        return {}
    hints = select_hints(state['question'], state.get('graph_retrieval_hints', []))
    if not hints:
        return {}
    # Recheck metadata scope. Graph names are advisory and may be stale; only
    # current original passages can enter the answer's evidence pack.
    if metadata_store is None:
        from src.api.routes_ingest import get_metadata_store
        metadata_store = get_metadata_store()
    docs = await metadata_store.list_documents()
    by_name = {}
    for doc in docs:
        by_name.setdefault(doc.filename, []).append(doc)
    terms, doc_ids = [], []
    for hint in hints:
        if any(len(by_name.get(name, [])) != 1 for name in hint['filenames']):
            continue
        matched = [d for name in hint['filenames'] for d in by_name.get(name, [])]
        if (len(matched) != len(hint['filenames']) or any(d.doc_id not in allowed
                or ('ALL' not in groups and not set(groups).intersection(d.acl_groups))
                or (state.get('dataset_id') and d.dataset_id != state['dataset_id'])
                or (state.get('edition_decisions', {}).get(d.doc_id, {}).get('source_revision')
                    not in (None, '', d.content_hash)) for d in matched)):
            continue
        terms.append(hint['term'])
        doc_ids.extend(d.doc_id for d in matched)
    doc_ids = list(dict.fromkeys(doc_ids))[:3]
    if not doc_ids:
        return {}
    from src.ingestion.embedder import embed_query
    query = state['question'] + '\nRelated document terms: ' + '; '.join(terms)
    vector = await asyncio.to_thread(embed_query, query)
    found = await asyncio.to_thread(vector_store.hybrid_search_reranked, vector=vector,
        text_query=query, user_groups=groups, top_k=settings.technical_followup_top_k,
        tier='medium', doc_ids=doc_ids)
    found = await asyncio.to_thread(vector_store.expand_figure_source_pages, found, groups,
        doc_ids, settings.technical_section_max_chars)
    # Do not elevate graph prose, unrelated results, or table-row summaries.
    needles = [re.sub(r'\W+', '', t.casefold()) for t in terms]
    from src.agent.quote_support import primary_text
    originals, used, seen = [], 0, set()
    for c in found:
        key = chunk_key(c)
        if (key in seen or c.metadata.doc_id not in doc_ids or c.metadata.content_type != 'text'
                or c.metadata.chunk_size_tier != 'medium'):
            continue
        text = re.sub(r'\W+', '', primary_text(c.text).casefold())
        if not any(term in text for term in needles) or used + len(c.text) > settings.technical_section_max_chars:
            continue
        originals.append(c); seen.add(key); used += len(c.text)
    if not originals:
        return {}
    return {'retrieved_chunks': originals,
        'technical_context_keys': list(dict.fromkeys(state.get('technical_context_keys', []) +
                                                     [repr(chunk_key(c)) for c in originals])),
        'graph_retrieval_trace': {'terms': terms, 'doc_ids': doc_ids, 'source_passages': len(originals), 'searches': 1}}
