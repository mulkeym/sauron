from unittest.mock import MagicMock
import pytest
from src.agent.profiles import AnswerProfile,snapshot
from src.agent.synthesizer import get_system_prompt,synthesis_question
from src.agent.strategies.technical import retrieve_technical
from src.retrieval.models import ChunkMetadata,RetrievedChunk


def test_diagnostic_starting_points_respect_selected_evidence_policy():
    question="I'm having problems with MPLS not working in SDWAN. what do I do?"
    for policy in ['partial','abstain']:
        profile=snapshot('test',1,AnswerProfile(name='Test',insufficient_evidence=policy))
        state={'question':question,'technical_intent':'troubleshooting','answer_profile':profile}
        prompt=get_system_prompt(profile,technical_intent='troubleshooting')
        assert ('documented diagnostic starting points' in prompt) == (policy=='partial')
        framed=synthesis_question(state)
        assert question in framed
        assert ('Answer the supported diagnostic part' in framed) == (policy=='partial')
    assert synthesis_question({'question':question,'technical_intent':'','answer_profile':profile}) == question


@pytest.mark.asyncio
async def test_technical_retrieval_recovers_original_evidence_for_a_figure_hit(monkeypatch):
    from src.agent.strategies import lookup,technical
    from src.ingestion import embedder
    from src.config import settings
    figure=RetrievedChunk(text='generated',score=1,metadata=ChunkMetadata(doc_id='d',filename='guide.pdf',
        doc_type='pdf',chunk_index=100,start_char=0,acl_groups=['team'],content_type='figure',page=39,figure_id='f'))
    original=figure.model_copy(update={'text':'Use a filter for diagnostics','metadata':figure.metadata.model_copy(
        update={'content_type':'text','figure_id':'','chunk_index':20})})
    monkeypatch.setattr(lookup,'retrieve_lookup',lambda *a:{'retrieved_chunks':[figure]})
    monkeypatch.setattr(embedder,'embed_query',lambda *a:[0.1])
    monkeypatch.setattr(settings,'technical_structure_enabled',False)
    vs=MagicMock()
    vs.expand_figure_source_pages.return_value=[figure,original]
    vs.hybrid_search_reranked.return_value=[]
    state={'question':'MPLS not working','user_groups':['team'],'allowed_doc_ids':['d']}
    result=await retrieve_technical(state,vs,'troubleshooting')
    assert original in result['retrieved_chunks']
    vs.expand_figure_source_pages.assert_called_once_with([figure],['team'],['d'],settings.technical_section_max_chars)
    assert result['technical_coverage']['followup_searches'] == 1


def test_technical_evidence_keeps_originals_ahead_of_bounded_derived_context(monkeypatch):
    from src.agent.synthesizer import build_evidence_pack
    from src.config import settings
    monkeypatch.setattr('src.agent.synthesizer._source_urls', lambda ids: {})
    monkeypatch.setattr('src.agent.synthesizer.native_figure_passages', lambda chunks, **kw: {})
    monkeypatch.setattr(settings, 'llm_max_context', 20000)
    def chunk(did, text, score, **metadata):
        return RetrievedChunk(text=text, score=score, metadata=ChunkMetadata(
            doc_id=did, filename=did + '.pdf', doc_type='pdf', chunk_index=0,
            start_char=0, acl_groups=['team'], **metadata))
    original = chunk('guide', 'Original diagnostic steps and restrictions. ' * 20, .3)
    figure = chunk('guide', 'Generated visual interpretation.', 1, content_type='figure', figure_id='f')
    graph = chunk('knowledge-graph', 'Generated graph context. ' * 300, .5)
    state = {'question': 'MPLS not working', 'technical_intent': 'troubleshooting',
             'allowed_doc_ids': ['guide'], 'retrieved_chunks': [graph, figure, original]}
    pack = build_evidence_pack(state)
    assert [c.source_kind for c in pack.citations] == ['document', 'derived']
    assert pack.citations[0].snippet == original.text
    assert 'Generated graph context' not in pack.model_context
    assert len(pack.aliases) == 2
    image_only = build_evidence_pack(dict(state, retrieved_chunks=[figure]))
    assert image_only.citations[0].figure_id == 'f'
    assert image_only.citations[0].source_kind == 'derived'
    assert any('Supplemental generated context' in w for w in pack.warnings)
    # Figure evidence stays available; technical synthesis omits generated graph
    # summaries when it has original source support.
    for other in [dict(state, technical_intent=''), dict(state, retrieved_chunks=[graph])]:
        assert 'Generated graph context' in build_evidence_pack(other).model_context


def test_general_diagnostics_need_citations_not_a_forced_verbatim_quote():
    from src.agent.quote_support import needs_quoted_support
    from src.agent.synthesizer import EvidencePack, finalize_answer
    from src.retrieval.models import Citation
    state = {'question': 'MPLS not working; what do I do?', 'technical_intent': 'troubleshooting',
             'answer_profile': snapshot('test', 1, AnswerProfile(name='Test'))}
    pack = EvidencePack(citations=[Citation(doc_id='guide', filename='guide.pdf', doc_type='pdf',
        chunk_index=0, evidence_id='E1', snippet='Inspect the debug state with show platform conditions.', relevance=1)])
    assert not needs_quoted_support(state)
    assert finalize_answer(state, 'Inspect the debug state [E1].', pack,
                           require_quotes=needs_quoted_support(state))['response_kind'] == 'answer'
    # Optional quotations and commands remain literal checks.
    for bad in ['"Reset the platform"', '`debug reset all`']:
        assert finalize_answer(state, bad + ' [E1].', pack)['validation_reason'] == 'unsupported_quote'
    for question in ['Which commands should I run?', 'Give step-by-step troubleshooting', 'Explain the arrow direction']:
        assert needs_quoted_support(dict(state, question=question))


@pytest.mark.parametrize('policy', ['partial', 'abstain'])
def test_failed_diagnostic_generation_returns_only_authorized_originals_under_partial_policy(monkeypatch, policy):
    from src.agent import synthesizer as sy, quote_support
    from src.retrieval.models import Citation
    state = {'question': 'MPLS not working', 'technical_intent': 'troubleshooting',
        'allowed_doc_ids': ['guide'], 'answer_profile': snapshot('test', 1, AnswerProfile(name='Test', insufficient_evidence=policy))}
    def cite(eid, did, text, kind='document', score=1):
        return Citation(doc_id=did, filename=did+'.pdf', doc_type='pdf', chunk_index=0,
            evidence_id=eid, snippet=text, source_kind=kind, relevance=score, page=39)
    original = cite('Eoriginal', 'guide', 'Document: guide.pdf\nSummary: Generated text.\n\nInspect diagnostic filters.')
    pack = sy.EvidencePack(context='source', aliases={'E1':'Eoriginal'}, citations=[original,
        cite('Ederived', 'guide', 'Generated interpretation.', 'derived', 2),
        cite('Edenied', 'private', 'Private passage.', score=3)])
    monkeypatch.setattr(sy, 'build_evidence_pack', lambda _: pack)
    monkeypatch.setattr(quote_support, 'retrieve_quote_support', lambda *a: None)
    generator = MagicMock(return_value='SAURON_STATUS: answer\n"Reset all devices" [E1].')
    monkeypatch.setattr(sy, 'generate', generator)
    result = sy.synthesize_answer(state)
    assert generator.call_count == 2
    assert 'Reset all devices' not in result['answer']
    if policy == 'partial':
        assert result['validation_reason'] == 'source_excerpt_fallback'
        assert result['citations'] == [original]
        assert 'Inspect diagnostic filters.' in result['answer']
        assert 'Generated' not in result['answer'] and 'Private' not in result['answer']
        assert 'not a verified diagnosis' in result['answer']
    else:
        assert result['response_kind'] == 'insufficient_evidence' and not result['citations']
    # Missing original support cannot be replaced by model-generated figure text.
    pack.citations = pack.citations[1:]
    assert sy.original_evidence_fallback(state, pack) is None


def test_final_reranking_cannot_discard_recovered_applicability_or_reorder_fallback(monkeypatch):
    from src.agent.synthesizer import build_evidence_pack, original_evidence_fallback
    from src.agent.state import chunk_key
    from src.config import settings
    monkeypatch.setattr('src.agent.synthesizer._source_urls', lambda ids: {})
    monkeypatch.setattr(settings, 'llm_max_context', 20000)
    def chunk(did, text, score, index=0):
        return RetrievedChunk(text=text, score=score, metadata=ChunkMetadata(doc_id=did,
            filename=did+'.pdf', doc_type='pdf', chunk_index=index, start_char=0, page=39, acl_groups=['team']))
    release = chunk('guide', 'MPLS feature requires release 17.11.1a.', .001)
    denied = chunk('private', 'Private applicability.', .001)
    noise = [chunk('other', 'Other troubleshooting tool ' + str(i), .9, i) for i in range(12)]
    state = {'question': 'MPLS not working', 'technical_intent': 'troubleshooting',
        'retrieved_chunks': noise + [release, denied], 'allowed_doc_ids': ['guide', 'other'],
        'technical_context_keys': [repr(chunk_key(release)), repr(chunk_key(denied))],
        'answer_profile': snapshot('test', 1, AnswerProfile(name='Test'))}
    pack = build_evidence_pack(state)
    assert pack.citations[0].snippet == release.text
    assert 'Private applicability' not in pack.context
    fallback = original_evidence_fallback(state, pack)
    assert fallback['citations'][0].doc_id == 'guide'
    assert 'Private applicability' not in fallback['answer']


def test_general_diagnostic_retry_does_not_prime_invented_literals(monkeypatch):
    from src.agent import synthesizer as sy, quote_support
    from src.retrieval.models import Citation
    state = {'question': 'MPLS not working', 'technical_intent': 'troubleshooting',
        'answer_profile': snapshot('test', 1, AnswerProfile(name='Test'))}
    source = Citation(doc_id='guide', filename='guide.pdf', doc_type='pdf', chunk_index=0,
        evidence_id='Eoriginal', snippet='Use diagnostic filtering to inspect MPLS packets.', relevance=1)
    pack = sy.EvidencePack(context=source.snippet, citations=[source], aliases={'E1':'Eoriginal'})
    monkeypatch.setattr(sy, 'build_evidence_pack', lambda _: pack)
    retrieval = MagicMock(side_effect=AssertionError('A prose overview does not need a new literal search'))
    monkeypatch.setattr(quote_support, 'retrieve_quote_support', retrieval)
    generator = MagicMock(side_effect=['SAURON_STATUS: answer\nRun `reset everything` [E1].',
        'SAURON_STATUS: answer\nDiagnostic filters help inspect MPLS traffic [E1].'])
    monkeypatch.setattr(sy, 'generate', generator)
    result = sy.synthesize_answer(state)
    assert result['response_kind'] == 'answer'
    assert result.get('validation_reason') != 'source_excerpt_fallback'
    assert result['citations'] == [source]
    prompt = generator.call_args.kwargs['system_prompt']
    assert 'three short prose bullets' in prompt and 'reset everything' not in prompt
    retrieval.assert_not_called()
