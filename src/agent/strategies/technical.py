"""Procedure and troubleshooting retrieval with one bounded coverage search."""
from __future__ import annotations
import asyncio
import re
from src.config import settings
from src.ingestion.technical_chunks import roles
from src.agent.state import chunk_key


def technical_intent(question):
    if re.search(r'\b(?:compare|difference|changed between)\b.*\b(?:versions?|revisions?|editions?|releases?)\b',question,re.I):
        return 'version_comparison'
    if re.search(r'\b(?:troubleshoot|diagnos\w*|not working|failing|fails|timeout|packet loss|tunnel(?: is)? down|cannot connect)\b',question,re.I):
        return 'troubleshooting'
    if re.search(r'\b(?:command|syntax|cli reference)\b',question,re.I) and not re.search(r'\b(?:steps|how to|how do|procedure)\b',question,re.I):
        return 'command_reference'
    if re.search(r'\b(?:how (?:do|can|should|to)|steps (?:to|for)|configuration steps|list[^\n]*steps|procedure|configure|deploy|install|roll back)\b',question,re.I):
        return 'procedure'
    if re.search(r'\b(?:architecture|topology|how .+ works?|explain .+ network)\b',question,re.I):
        return 'architecture'
    return ''


def retrieval_subject(question):
    """Keep output-format instructions out of semantic search and reranking."""
    if not re.search(r'\bmermaid\b', question, re.I):
        return question
    subject = re.sub(r'\b(?:in|as|using)\s+(?:a\s+)?mermaid(?:\s+(?:format|syntax|code|diagram))?\b', '', question, flags=re.I)
    subject = re.sub(r'^(?:please\s+)?(?:draw|render|create|generate|show)(?:\s+me)?\s+', '', subject, flags=re.I)
    return ' '.join(subject.split()).strip(' ?.') or question


def retrieval_question(state):
    previous=[m.get('content','') for m in state.get('conversation', [])[-8:]
              if m.get('role')=='user' and isinstance(m.get('content'),str)]
    return ('User-reported context: '+'\n'.join(previous)[-4000:]+'\nCurrent question: ' if previous else '')+retrieval_subject(state['question'])


def coverage(chunks, intent):
    expected=(['symptoms','causes','diagnostics','interpretation','correction'] if intent=='troubleshooting'
              else ['prerequisites','steps','warnings','verification','rollback'])
    support={name:[] for name in expected}
    for chunk in chunks:
        if chunk.metadata.content_type=='figure' or chunk.metadata.doc_id in ('knowledge-graph','map-reduce','metadata-context'):
            continue  # Connectivity/derived prose cannot establish procedural steps.
        found=roles(chunk.text+'\n'+chunk.metadata.section_path)
        for name in expected:
            if name in found:support[name].append({'doc_id':chunk.metadata.doc_id,'chunk_index':chunk.metadata.chunk_index})
    return {'supported':support, 'missing':[name for name,spans in support.items() if not spans],
            'method':'source cues; not semantic entailment verification'}


async def retrieve_technical(state, vector_store, intent):
    from src.agent.strategies.lookup import retrieve_lookup
    from src.ingestion.embedder import embed_query
    result=await asyncio.to_thread(retrieve_lookup,state,vector_store)
    chunks=result.get('retrieved_chunks',[])
    before_expansion = {chunk_key(c) for c in chunks}
    if state.get('allowed_doc_ids') and any(c.metadata.content_type == 'figure' for c in chunks):
        chunks=await asyncio.to_thread(vector_store.expand_figure_source_pages, chunks,
            state.get('user_groups',[]), state['allowed_doc_ids'], settings.technical_section_max_chars)
    result['technical_context_keys'] = [repr(chunk_key(c)) for c in chunks if chunk_key(c) not in before_expansion]
    report=coverage(chunks,intent)
    calls=0
    if report['missing'] and state.get('allowed_doc_ids'):
        query=retrieval_question(state)+'\nDocumentation for: '+', '.join(report['missing'])
        vector=await asyncio.to_thread(embed_query,query)
        extra=await asyncio.to_thread(vector_store.hybrid_search_reranked,
            vector=vector,text_query=query,user_groups=state.get('user_groups',[]),
            top_k=settings.technical_followup_top_k,tier='medium',doc_ids=state['allowed_doc_ids'])
        seen={chunk_key(c) for c in chunks}
        chunks+= [c for c in extra if chunk_key(c) not in seen]
        calls=1
        if settings.technical_structure_enabled:
            chunks=await asyncio.to_thread(vector_store.expand_sections,chunks,state.get('user_groups',[]),
                state['allowed_doc_ids'],settings.technical_section_max_chars)
        report=coverage(chunks,intent)
    report['followup_searches']=calls
    result['retrieved_chunks']=chunks
    result['technical_coverage']=report
    result['technical_intent']=intent
    if report['missing']:
        result['warnings']=list(state.get('warnings',[]))+[
            'Automatic coverage check did not identify '+', '.join(report['missing'])+
            ' in the retrieved passages. This is a keyword-based check, not a finding that the document lacks them.']
    return result


GUIDANCE = {
    'procedure': 'Organize an answer by prerequisites, steps, warnings, verification and rollback as relevant to the question.',
    'troubleshooting': 'Separate user-reported symptoms from source facts. Distinguish possible causes, diagnostic checks, outcomes and corrective actions.',
    'command_reference': 'Include documented syntax, applicability, cautions and required modes.',
    'architecture': 'Explain the documented components and relationships.',
    'version_comparison': 'Label each edition separately when comparing its instructions.',
}
