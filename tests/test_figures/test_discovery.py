from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.figures.service import diagram_discovery_question
from src.agent.graph import create_agent_graph
from src.agent.state import QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata
from src.retrieval.query_scope import QueryScope


@pytest.mark.parametrize('q', ['list the diagrams for sdwan', 'can you show me a architectual diagram for sdwan?',
    'Please find the network topology diagram', 'do you have diagrams?', 'what diagrams do you have?', 'show a Venn diagram',
    'do we have any figures related to sdwan?\n', 'Do I have any diagrams for SD-WAN?',
    'Have we got diagrams for sdwan?', 'Are any diagrams available for sdwan?',
    'Which network diagrams are available?', 'Do the documents contain any diagrams?',
    'can you show me images for sdwan?'])
def test_discovery_intent(q):
    assert diagram_discovery_question(q)


@pytest.mark.parametrize('q', ['Explain the arrows in this diagram', 'Compare these topology diagrams',
    'List the steps in the diagram', 'What does this diagram mean?', 'How do I configure SD-WAN?',
    'Show the differences between these diagrams', 'Describe the topology'])
def test_explanation_remains_normal_answer(q):
    assert not diagram_discovery_question(q)


@pytest.mark.asyncio
@pytest.mark.parametrize('question', ['list the diagrams for sdwan', 'do we have any figures related to sdwan?\n'])
@pytest.mark.parametrize('has_access', [True, False])
async def test_diagram_discovery_bypasses_llm_catalog_and_reranking(monkeypatch, has_access, question):
    from src.agent import graph as graph_module
    from src.agent import synthesizer
    from src.figures import service
    from src.retrieval import query_scope
    from src.generation import reasoning
    scope = QueryScope(('d1',) if has_access else (), 'revision')
    monkeypatch.setattr(query_scope, 'resolve_query_scope', AsyncMock(return_value=scope))
    classifier = AsyncMock(side_effect=AssertionError('No model classification for diagram browsing'))
    monkeypatch.setattr(graph_module, '_classify_node_factory', lambda registry: classifier)
    monkeypatch.setattr(graph_module, '_lookup_then_structured', AsyncMock(side_effect=AssertionError('No catalog/text lookup')))
    monkeypatch.setattr(synthesizer, 'generate', MagicMock(side_effect=AssertionError('No generated answer for a figure list')))
    monkeypatch.setattr(reasoning, 'answer_generation_kwargs', MagicMock(side_effect=AssertionError('No provider capability request')))
    chunk = RetrievedChunk(text='Diagram: branch hub topology', score=.03, metadata=ChunkMetadata(
        doc_id='d1', filename='sdwan.pdf', doc_type='pdf', acl_groups=['team'],
        chunk_index=1, start_char=0, content_type='figure', figure_id='p2-fig-001', page=2))
    async def search(*args, **kwargs):
        assert kwargs['allowed_doc_ids'] == list(scope.doc_ids)
        assert args[1] == ['team']
        return [chunk, chunk] if has_access else []
    monkeypatch.setattr(service, 'search_chunks', search)
    monkeypatch.setattr(service, 'authorized_figure', AsyncMock(return_value=(SimpleNamespace(), {})))
    monkeypatch.setattr(service, 'reference', lambda *a: {'available': True})
    vs = MagicMock()
    graph = create_agent_graph(vs, MagicMock(), MagicMock())
    result = await graph.ainvoke({'question':question, 'user_groups':['team'], 'retrieved_chunks':[]})
    classifier.assert_not_called()
    vs.rerank_chunks.assert_not_called()
    assert result['diagram_discovery'] and result['query_type'] == QueryType.LOOKUP
    if has_access:
        assert len(result['citations']) == 1
        assert result['citations'][0].figure_id == 'p2-fig-001'
        assert 'Found 1 stored figure' in result['answer']
    else:
        assert result['citations'] == [] and 'No stored figures' in result['answer']


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,question,expected', [
    ('auto', 'Explain SD-WAN for government to me', True),
    ('requested', 'Explain SD-WAN for government to me', False),
    ('off', 'Explain this SD-WAN diagram', False),
    ('requested', 'Explain this SD-WAN diagram', True),
])
async def test_normal_explanations_search_figures_according_to_policy(monkeypatch, mode, question, expected):
    from src.agent import graph as graph_module
    from src.agent.profiles import AnswerProfile
    from src.agent.strategies import graph_sources
    from src.figures import service
    from src.retrieval import query_scope
    scope = QueryScope(('d1',), 'revision')
    monkeypatch.setattr(query_scope, 'resolve_query_scope', AsyncMock(return_value=scope))
    monkeypatch.setattr(graph_module, '_classify_node_factory', lambda registry: AsyncMock(return_value={'query_type':QueryType.LOOKUP}))
    monkeypatch.setattr(graph_module, '_lookup_then_structured', AsyncMock(return_value={'retrieved_chunks':[]}))
    monkeypatch.setattr(graph_sources, 'retrieve_graph_sources', AsyncMock(return_value={}))
    chunk = RetrievedChunk(text='Government SD-WAN architecture', score=.9, metadata=ChunkMetadata(
        doc_id='d1', filename='sdwan.pdf', doc_type='pdf', acl_groups=['team'],
        chunk_index=1, start_char=0, content_type='figure', figure_id='p2-fig-001', page=2))
    search = AsyncMock(return_value=[chunk])
    monkeypatch.setattr(service, 'search_chunks', search)
    profile = {'config':AnswerProfile(name='Test', images=mode, graph_enrichment=False).model_dump()}
    graph = create_agent_graph(MagicMock(), MagicMock(), MagicMock(), include_synthesize=False)
    result = await graph.ainvoke({'question':question, 'user_groups':['team'], 'retrieved_chunks':[], 'answer_profile':profile})
    assert bool(search.await_count) == expected
    if expected:
        assert search.await_args.kwargs['allowed_doc_ids'] == ['d1']
        assert result['retrieved_chunks'][0].metadata.figure_id == 'p2-fig-001'
