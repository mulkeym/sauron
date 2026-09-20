from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.profiles import AnswerProfile, snapshot
from src.agent.quote_support import retrieve_quote_support, unsupported_quotes, needs_quoted_support
from src.agent.synthesizer import EvidencePack, finalize_answer, synthesize_answer
from src.retrieval.models import Citation, ChunkMetadata, RetrievedChunk


def citation(eid='Etext', text='If the status is not READY within 90 seconds, restore the configuration.', **kw):
    return Citation(evidence_id=eid, doc_id='doc', filename='guide.pdf', doc_type='pdf',
                    chunk_index=1, snippet=text, relevance=1, **kw)


def test_quote_must_match_its_own_primary_citation_not_another_source():
    source = citation()
    figure = citation('Efigure', 'A screen with a timeline.', source_kind='derived')
    body = '"If the status is not READY within 90 seconds" [Efigure].'
    assert unsupported_quotes(body, [source, figure])
    assert not unsupported_quotes(body.replace('Efigure', 'Etext'), [source, figure])
    assert unsupported_quotes(body.replace('Efigure', 'Etext').replace('not READY', 'READY'), [source])
    assert unsupported_quotes(body.replace('Efigure', 'Etext').replace('90', '60'), [source])
    assert unsupported_quotes('"Jade-B to Jade-A" [Etext]', [citation(text='Jade-A to Jade-B')])


def test_quote_normalizes_pdf_layout_but_not_case_or_generated_summaries():
    source = citation(text='IfthestatusisnotREADYwithin90seconds,restoretheconfiguration.')
    assert not unsupported_quotes('“If the status is not READY\nwithin 90 seconds” [Etext]', [source])
    assert unsupported_quotes('"If the status is not ready within 90 seconds" [Etext]', [source])
    assert unsupported_quotes('"not READY within 9 0 seconds" [Etext]', [source])
    generated = citation(text='Document: guide.pdf\nSummary: A secure federal network.\n\nOriginal unrelated text.')
    assert unsupported_quotes('"A secure federal network" [Etext]', [generated])
    assert not unsupported_quotes('The "special" feature is documented [Etext].', [source], 'what is special?')


def test_failed_quote_returns_no_claim_or_citation_and_repair_cannot_drop_quotes():
    p = EvidencePack(citations=[citation()])
    result = finalize_answer({}, 'The answer is "READY within 60 seconds" [Etext].', p)
    assert result['validation_reason'] == 'unsupported_quote'
    assert not result['citations'] and '60 seconds' not in result['answer']
    assert finalize_answer({}, 'Restore within 60 seconds [Etext].', p,
                           require_quotes=True)['validation_reason'] == 'unsupported_quote'


def test_direction_and_procedures_need_original_quoted_support():
    assert needs_quoted_support({'question': 'What do the two lines represent?'})
    assert needs_quoted_support({'question': 'How do I configure this?', 'technical_intent': 'procedure'})
    assert not needs_quoted_support({'question': 'What is special about this product?'})


def test_quote_retrieval_checks_current_acl_dataset_and_revision(monkeypatch):
    from src.api import routes_ingest
    source = citation()
    p = EvidencePack(citations=[source])
    failures = [{'quote': 'not READY within 90 seconds', 'evidence_ids': ['Etext']}]
    state = {'question': 'when to restore?', 'user_groups': ['team'], 'allowed_doc_ids': ['doc'],
             'dataset_id': 1, 'edition_decisions': {'doc': {'source_revision': 'saved'}}}
    meta = SimpleNamespace(content_hash='saved', acl_groups=['team'], dataset_id=1)
    store = SimpleNamespace(get_document=AsyncMock(return_value=meta))
    vs = MagicMock()
    chunk = RetrievedChunk(text=source.snippet, score=.5, metadata=ChunkMetadata(
        doc_id='doc', filename='guide.pdf', doc_type='pdf', chunk_index=1, start_char=0, acl_groups=['team']))
    vs.read_document_page.return_value = ([chunk], False)
    monkeypatch.setattr(routes_ingest, 'get_metadata_store', lambda: store)
    monkeypatch.setattr(routes_ingest, 'get_vector_store', lambda: vs)
    repaired = retrieve_quote_support(state, p, failures)
    assert repaired['retrieved_chunks'][0].text == source.snippet
    assert repaired['edition_decisions'] == state['edition_decisions']
    vs.read_document_page.assert_called_once_with('doc', ['team'], limit=256)
    for overrides in ({'content_hash': 'changed'}, {'acl_groups': ['other']}, {'dataset_id': 2}):
        vs.reset_mock()
        store.get_document.return_value = SimpleNamespace(**{**vars(meta), **overrides})
        assert retrieve_quote_support(state, p, failures) is None
        vs.read_document_page.assert_not_called()
    assert retrieve_quote_support({**state, 'allowed_doc_ids': []}, p, failures) is None


def test_quote_repair_regenerates_against_new_evidence_never_relabels_old_answer(monkeypatch):
    from src.agent import synthesizer, quote_support
    state = {'question': 'when to restore?', 'answer_profile': snapshot('test', 1, AnswerProfile(name='test'))}
    old = EvidencePack(context='old', model_context='[E1] A screen.',
        citations=[citation('Efigure', 'A screen.', source_kind='derived')], aliases={'E1': 'Efigure'})
    new = EvidencePack(context='new', model_context='[E1] ' + citation().snippet,
        citations=[citation()], aliases={'E1': 'Etext'})
    monkeypatch.setattr(synthesizer, 'build_evidence_pack', MagicMock(side_effect=[old, new]))
    monkeypatch.setattr(quote_support, 'retrieve_quote_support', lambda *a: {**state, 'repaired': True})
    generator = MagicMock(side_effect=[
        'SAURON_STATUS: answer\n"not READY within 90 seconds" [E1].',
        'SAURON_STATUS: answer\n"If the status is not READY within 90 seconds" [E1].'])
    monkeypatch.setattr(synthesizer, 'generate', generator)
    result = synthesize_answer(state)
    assert result['response_kind'] == 'answer'
    assert [c.evidence_id for c in result['citations']] == ['Etext']
    assert generator.call_count == 2
    assert generator.call_args_list[0].kwargs['user_prompt'] != generator.call_args_list[1].kwargs['user_prompt']
    assert 'A screen.' not in generator.call_args_list[1].kwargs['user_prompt']


def test_inline_commands_need_literal_original_support_even_if_user_supplies_them():
    source = citation(text='Run show system status to check FIPS.')
    assert not unsupported_quotes('Run `show system status` [Etext].', [source])
    assert unsupported_quotes('Run `show certificate signing-request decoded` [Etext].', [source])
    assert unsupported_quotes('Run `show bogus` [Etext].', [source], 'Can I run show bogus?')


@pytest.mark.parametrize('policy', ['partial', 'abstain'])
@pytest.mark.parametrize('clarification', ['when_needed', 'answer_with_caveats'])
@pytest.mark.parametrize('fields', [[], ['software_version']])
@pytest.mark.parametrize('repair_kind', ['format', 'quotation'])
def test_repair_keeps_selected_policy_and_clarification_contract(policy, clarification, fields, repair_kind):
    from src.agent.synthesizer import get_system_prompt, FORMAT_REPAIR, QUOTE_REPAIR
    profile = AnswerProfile(name='test', insufficient_evidence=policy,
                            clarification=clarification, clarification_fields=fields)
    prompt = get_system_prompt(snapshot('test', 1, profile), technical_intent='procedure',
                               repair=FORMAT_REPAIR if repair_kind == 'format' else QUOTE_REPAIR)
    assert prompt.count('Response format') == 1
    assert ('Answer the supported part' in prompt) == (policy == 'partial')
    assert ('Answer only if evidence supports the complete requested answer' in prompt) == (policy == 'abstain')
    assert ('SAURON_STATUS: clarification' in prompt) == (clarification == 'when_needed' and bool(fields))
    assert 'If no supported quotation answers the question' not in prompt
    assert 'include quoted original support' in prompt


@pytest.mark.parametrize("intent, expected", [("", "answer"), ("procedure", "insufficient_evidence"), ("command_reference", "insufficient_evidence")])
def test_repair_preserves_original_quoted_support_requirement(monkeypatch, intent, expected):
    from src.agent import synthesizer, quote_support
    state = {'question': 'What is documented?', 'technical_intent': intent, 'answer_profile': snapshot('test', 1, AnswerProfile(name='test'))}
    evidence = EvidencePack(context='source', citations=[citation()], aliases={'E1': 'Etext'})
    monkeypatch.setattr(synthesizer, 'build_evidence_pack', lambda _: evidence)
    monkeypatch.setattr(quote_support, 'retrieve_quote_support', lambda *args: None)
    generator = MagicMock(side_effect=['SAURON_STATUS: answer\n"READY within 60 seconds" [E1].',
                                      'SAURON_STATUS: answer\nRestore if readiness fails within 90 seconds [E1].'])
    monkeypatch.setattr(synthesizer, 'generate', generator)
    assert synthesize_answer(state)['response_kind'] == expected
    assert generator.call_count == 2


def test_verified_inline_literals_count_as_original_support():
    from src.agent.quote_support import has_source_quote
    source = citation(text='The filename follows the format date-time-admin-tech.tar.gz.')
    answer = 'The output is `date-time-admin-tech.tar.gz` [Etext].'
    assert has_source_quote(answer, citations=[source])
    assert finalize_answer({}, answer, EvidencePack(citations=[source]), require_quotes=True)['response_kind'] == 'answer'
    for invalid in ['The file is `invented.tar.gz` [Etext].', 'Open `guide.pdf` [Etext].',
                    '`date-time-admin-tech.tar.gz` [Eother].', '```text\ndate-time-admin-tech.tar.gz\n```\n[Etext]']:
        assert not has_source_quote(invalid, citations=[source])
    generated = citation(text=source.snippet, source_kind='derived')
    assert not has_source_quote(answer, citations=[generated])
