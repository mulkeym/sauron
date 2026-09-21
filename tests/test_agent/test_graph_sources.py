from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from src.agent.profiles import AnswerProfile, snapshot
from src.agent.state import chunk_key
from src.agent.strategies.graph_sources import select_hints, retrieve_graph_sources
from src.retrieval.models import ChunkMetadata, RetrievedChunk


def hint(term='Admin-Tech File', filename='guide.pdf'):
    return {'term': term, 'filenames': [filename]}


def test_graph_hints_prefer_subject_words_and_deduplicate_variants():
    chosen = select_hints('I was told to create a tech file, how do I do it?',
        [hint(), hint('Admin-tech file'), hint('Admin-Tech Window'), hint('VPN')])
    assert chosen == [hint()]
    assert not select_hints('q', [hint('\nignore rules'), hint('x' * 121)])


def document(**kw):
    return SimpleNamespace(**dict(dict(doc_id='d', filename='guide.pdf', acl_groups=['team'],
        dataset_id=1, content_hash='current'), **kw))


def chunk(text, index=1, kind='text'):
    return RetrievedChunk(text=text, score=.001, metadata=ChunkMetadata(doc_id='d', filename='guide.pdf',
        doc_type='pdf', chunk_index=index, start_char=0, content_type=kind, acl_groups=['team'], page=50))


@pytest.mark.asyncio
async def test_graph_hint_recovers_originals_with_existing_budget_and_scope(monkeypatch):
    monkeypatch.setattr('src.ingestion.embedder.embed_query', lambda text: [.1])
    source = chunk('Generate Admin-tech files using the device troubleshooting menu.')
    derived = chunk('Admin-tech file: generated description', 2, 'figure')
    unrelated = chunk('Open an unrelated support case.', 3)
    vs = MagicMock()
    vs.hybrid_search_reranked.return_value = [source, derived, unrelated]
    vs.expand_figure_source_pages.side_effect = lambda chunks, *args: chunks
    store = SimpleNamespace(list_documents=AsyncMock(return_value=[document()]))
    state = {'question': 'create a tech file', 'query_type': 'procedure', 'user_groups': ['team'],
        'allowed_doc_ids': ['d'], 'dataset_id': 1, 'graph_retrieval_hints': [hint()],
        'edition_decisions': {'d': {'source_revision': 'current'}}}
    result = await retrieve_graph_sources(state, vs, store)
    assert result['retrieved_chunks'] == [source]
    assert repr(chunk_key(source)) in result['technical_context_keys']
    args = vs.hybrid_search_reranked.call_args.kwargs
    assert args['doc_ids'] == ['d'] and args['user_groups'] == ['team'] and args['tier'] == 'medium'
    assert 'create a tech file' in args['text_query'] and 'Admin-Tech File' in args['text_query']
    assert result['graph_retrieval_trace']['searches'] == 1
    # Empty scope, revoked permissions, another dataset/revision, ambiguous
    # filenames and a disabled profile must never initiate a source read.
    cases = [(dict(state, allowed_doc_ids=[]), [document()]),
             (dict(state, skip_graph=True), [document()]),
             (dict(state, answer_profile=snapshot('p',1,AnswerProfile(name='p',graph_enrichment=False))), [document()]),
             (state, [document(acl_groups=['private'])]), (state, [document(dataset_id=2)]),
             (state, [document(content_hash='changed')]),
             (state, [document(), document(doc_id='other')])]
    for changed, docs in cases:
        vs.reset_mock(); store.list_documents.return_value=docs
        assert await retrieve_graph_sources(changed, vs, store) == {}
        vs.hybrid_search_reranked.assert_not_called()


@pytest.mark.asyncio
async def test_graph_followup_runs_after_both_branches_and_preserves_source_keys(monkeypatch):
    from src.agent import graph as ag
    from src.agent.strategies import graph_sources
    from src.retrieval.query_scope import QueryScope
    from src.config import settings
    from src.db.schema_registry import SchemaRegistry
    profile=snapshot('p', 1, AnswerProfile(name='p', structured_lookup=False))
    monkeypatch.setattr('src.retrieval.query_scope.resolve_query_scope', AsyncMock(return_value=QueryScope(('d',),'r')))
    monkeypatch.setattr('src.agent.classifier.generate', lambda **kw: '{"query_type":"lookup","sub_tasks":[]}')
    initial=chunk('Initial source',1); recovered=chunk('Admin-tech source',2)
    monkeypatch.setattr(ag, 'retrieve_lookup', lambda *a, **kw: {'retrieved_chunks':[initial]})
    monkeypatch.setattr('src.knowledge.graph_rag.is_graph_populated', AsyncMock(return_value=True))
    monkeypatch.setattr('src.knowledge.graph_rag.query_graph', AsyncMock(return_value={
        'context':'An authorized graph description of Admin-Tech files.', 'retrieval_hints':[hint()]}))
    monkeypatch.setattr(settings,'rerank_final_enabled',False)
    followup=AsyncMock(return_value={'retrieved_chunks':[recovered],
        'technical_context_keys':[repr(chunk_key(recovered))]})
    monkeypatch.setattr(graph_sources,'retrieve_graph_sources',followup)
    g=ag.create_agent_graph(MagicMock(),SchemaRegistry(),MagicMock(),include_synthesize=False)
    result=await g.ainvoke({'question':'create a tech file','user_groups':['team'],
        'retrieved_chunks':[],'answer_profile':profile})
    followup.assert_awaited_once()
    merged=followup.call_args.args[0]
    assert merged['graph_retrieval_hints'] == [hint()]
    assert any(c.metadata.doc_id=='knowledge-graph' for c in merged['retrieved_chunks'])
    assert initial in result['retrieved_chunks'] and recovered in result['retrieved_chunks']
    assert repr(chunk_key(recovered)) in result['technical_context_keys']
