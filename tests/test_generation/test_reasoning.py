import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from src.config import Settings, settings
from src.generation import llm_client as llm, reasoning as r


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setattr(settings, 'vllm_base_url', 'https://openrouter.ai/api/v1')
    monkeypatch.setattr(settings, 'vllm_model_name', 'google/gemma-4-26b-a4b-it')
    monkeypatch.setattr(settings, 'llm_reasoning_adapter', 'auto')
    monkeypatch.setattr(settings, 'llm_answer_thinking', 'default')
    monkeypatch.setattr(settings, 'llm_answer_temperature', 0.1)
    r._cache.clear()
    yield
    r._cache.clear()


def metadata(monkeypatch, *, supported=True, mandatory=False):
    get = MagicMock(return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'data': [{
        'id': settings.vllm_model_name, 'supported_parameters': ['reasoning'] if supported else [],
        'reasoning': {'mandatory': mandatory, 'default_enabled': False}}]}))
    monkeypatch.setattr(r.requests, 'get', get)
    return get


def post(monkeypatch, *, content='SAURON_STATUS: answer\nFinal [E1].', finish='stop', **fields):
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'choices': [
        {'message': {'content': content, **fields}, 'finish_reason': finish}]})
    mock = MagicMock(return_value=response)
    monkeypatch.setattr(llm, 'post_json', mock)
    return mock


@pytest.mark.parametrize('mode, enabled', [('enabled', True), ('disabled', False)])
def test_real_outgoing_openrouter_request_uses_advertised_control(monkeypatch, mode, enabled):
    get = metadata(monkeypatch)
    send = post(monkeypatch)
    answer = llm.generate('Rules', 'Question', max_tokens=32768, reasoning_mode=mode)
    payload = send.call_args.kwargs['json']
    assert payload['reasoning'] == {'enabled': enabled, 'exclude': True}
    assert payload['provider'] == {'require_parameters': True}
    assert 'chat_template_kwargs' not in payload
    assert payload['max_tokens'] == 32768  # Not silently replaced by SQL's 4096.
    assert payload['messages'][0]['content'] == 'Rules'  # No "think" prompt injection.
    assert answer == 'SAURON_STATUS: answer\nFinal [E1].'
    assert get.call_count == 1


def test_provider_default_omits_reasoning_and_metadata_lookup(monkeypatch):
    get = metadata(monkeypatch)
    send = post(monkeypatch)
    llm.generate('s', 'q')
    assert not {'reasoning', 'provider', 'chat_template_kwargs'}.intersection(send.call_args.kwargs['json'])
    get.assert_not_called()
    assert Settings(_env_file=None).llm_answer_thinking == 'default'


def test_metadata_cache_is_model_specific_and_expires(monkeypatch):
    get = metadata(monkeypatch)
    r.reasoning_capability()
    r.reasoning_capability()
    assert get.call_count == 1
    assert not r.reasoning_capability(model='different')['supported']
    assert get.call_count == 2
    monkeypatch.setattr(r.time, 'monotonic', lambda: 10**15)
    r.reasoning_capability()
    assert get.call_count == 3


@pytest.mark.parametrize('supported, mandatory, mode', [(False, False, 'enabled'), (True, True, 'disabled')])
def test_unsupported_control_fails_before_post(monkeypatch, supported, mandatory, mode):
    metadata(monkeypatch, supported=supported, mandatory=mandatory)
    send = post(monkeypatch)
    with pytest.raises(llm.LLMError):
        llm.generate('s', 'q', reasoning_mode=mode)
    send.assert_not_called()


def test_unavailable_metadata_does_not_pretend_enablement(monkeypatch):
    monkeypatch.setattr(r.requests, 'get', MagicMock(side_effect=requests.Timeout()))
    send = post(monkeypatch)
    with pytest.raises(llm.LLMError, match='Cannot verify'):
        llm.generate('s', 'q', reasoning_mode='enabled')
    send.assert_not_called()


@pytest.mark.parametrize('url', ['https://api.openai.com/v1', 'https://openrouter.ai.example/v1', 'http://localhost:8000/v1'])
def test_unknown_auto_endpoints_never_get_a_guessed_extension(monkeypatch, url):
    monkeypatch.setattr(settings, 'vllm_base_url', url)
    send = post(monkeypatch)
    with pytest.raises(llm.LLMError, match='No verified'):
        llm.generate('s', 'q', reasoning_mode='enabled')
    send.assert_not_called()
    llm.generate('s', 'q')  # Backwards-compatible omission.
    assert 'chat_template_kwargs' not in send.call_args.kwargs['json']


@pytest.mark.parametrize('mode, enabled', [('enabled', True), ('disabled', False)])
def test_explicit_vllm_adapter_sets_template_control(monkeypatch, mode, enabled):
    monkeypatch.setattr(settings, 'vllm_base_url', 'http://localhost:8000/v1')
    monkeypatch.setattr(settings, 'llm_reasoning_adapter', 'vllm_template')
    send = post(monkeypatch)
    llm.generate('s', 'q', reasoning_mode=mode)
    assert send.call_args.kwargs['json']['chat_template_kwargs'] == {'enable_thinking': enabled}
    assert not r.reasoning_capability()['default_enabled']
    assert 'not automatically verified' in r.reasoning_capability()['detail']


def test_provider_rejection_does_not_retry_without_reasoning(monkeypatch):
    metadata(monkeypatch)
    send = MagicMock(return_value=SimpleNamespace(text='Unsupported reasoning parameter',
        raise_for_status=MagicMock(side_effect=requests.HTTPError('400'))))
    monkeypatch.setattr(llm, 'post_json', send)
    with pytest.raises(llm.LLMError, match='Unsupported reasoning'):
        llm.generate('s', 'q', reasoning_mode='enabled')
    assert send.call_count == 1
    assert send.call_args.kwargs['json']['reasoning']['enabled']


@pytest.mark.parametrize('content', [None, '', '<think>private</think>', '<think>unclosed', '<|channel>thoughtprivate<channel|>'])
def test_never_fall_back_to_internal_reasoning(monkeypatch, content):
    post(monkeypatch, content=content, reasoning='private', reasoning_content='private', text='private')
    with pytest.raises(llm.LLMError, match='no final answer'):
        llm.generate('s', 'q')


@pytest.mark.parametrize('content', ['', 'partial answer'])
def test_token_exhaustion_is_provider_error_not_evidence_abstention(monkeypatch, content):
    post(monkeypatch, content=content, finish='length', reasoning='private')
    with pytest.raises(llm.LLMError, match='token budget exhausted'):
        llm.generate('s', 'q')


def test_final_only_content_preserves_status_json_and_citations(monkeypatch):
    post(monkeypatch, content='<think>private</think>SAURON_STATUS: answer\nQuoted [E1].', reasoning='private')
    assert llm.generate('s', 'q') == 'SAURON_STATUS: answer\nQuoted [E1].'
    assert llm.parse_json_response('<|channel>thoughtprivate<channel|>{"ok":true}') == {'ok': True}
    with pytest.raises(ValueError, match='No final'):
        llm.parse_json_response('<think>{"private":true}</think>')


@pytest.mark.parametrize('opening, closing', list(r.FinalTextFilter.pairs.items()))
def test_stream_filter_handles_every_split_delimiter_and_unclosed_tail(opening, closing):
    raw = 'A' + opening + 'PRIVATE' + closing + 'Final [E1].'
    for split in range(len(raw) + 1):
        parser = r.FinalTextFilter()
        assert parser.feed(raw[:split]) + parser.feed(raw[split:]) + parser.feed('', final=True) == 'AFinal [E1].'
    parser = r.FinalTextFilter()
    assert ''.join(parser.feed(c) for c in raw) + parser.feed('', final=True) == 'AFinal [E1].'
    parser = r.FinalTextFilter()
    assert parser.feed(opening + 'PRIVATE') + parser.feed('', final=True) == ''


def test_stream_payload_and_private_fields_are_separate(monkeypatch):
    metadata(monkeypatch)
    chunks = [{'choices': [{'delta': {'reasoning': 'PRIVATE', 'reasoning_content': 'PRIVATE'}}]}]
    chunks += [{'choices': [{'delta': {'content': c}}]} for c in '<think>PRIVATE</think>Final [E1].']
    chunks += [{'choices': [], 'usage': {}}]
    response = SimpleNamespace(raise_for_status=lambda: None, close=MagicMock(),
        iter_lines=lambda **kw: iter(['data: ' + json.dumps(x) for x in chunks] + ['data: [DONE]']))
    send = MagicMock(return_value=response)
    monkeypatch.setattr(llm.requests, 'post', send)
    assert ''.join(llm.generate_stream('s', 'q', reasoning_mode='enabled')) == 'Final [E1].'
    assert send.call_args.kwargs['json']['reasoning'] == {'enabled': True, 'exclude': True}
    response.close.assert_called_once()


@pytest.mark.parametrize('mode', ['default', 'enabled', 'disabled'])
def test_answer_mode_reaches_both_attempts_not_unrelated_calls(monkeypatch, mode):
    from src.agent import synthesizer
    from src.retrieval.models import Citation
    monkeypatch.setattr(settings, 'llm_answer_thinking', mode)
    monkeypatch.setattr(settings, 'llm_answer_temperature', 1.0)
    citation = Citation(evidence_id='Eknown', doc_id='d', filename='f', doc_type='pdf', chunk_index=0, snippet='fact', relevance=1)
    pack = synthesizer.EvidencePack(context='[E1] fact', citations=[citation], aliases={'E1':'Eknown'})
    monkeypatch.setattr(synthesizer, 'build_evidence_pack', lambda state: pack)
    gen = MagicMock(side_effect=['SAURON_STATUS: answer', 'SAURON_STATUS: answer\nfact [E1].'])
    monkeypatch.setattr(synthesizer, 'generate', gen)
    assert synthesizer.synthesize_answer({'question':'q'})['response_kind'] == 'answer'
    assert gen.call_count == 2
    assert all(c.kwargs.get('reasoning_mode', 'default') == mode for c in gen.call_args_list)
    assert all(c.kwargs['temperature'] == 1.0 for c in gen.call_args_list)
    send = post(monkeypatch)
    llm.generate('classification', 'q')
    assert 'reasoning' not in send.call_args.kwargs['json']
    assert send.call_args.kwargs['json']['temperature'] == 0.1


def test_stream_empty_length_is_error_without_leaking_reasoning(monkeypatch):
    response = SimpleNamespace(raise_for_status=lambda: None, close=lambda: None,
        iter_lines=lambda **kw: iter(['data: ' + json.dumps({'choices':[{'delta': {'reasoning':'PRIVATE'}, 'finish_reason':'length'}]}), 'data: [DONE]']))
    monkeypatch.setattr(llm.requests, 'post', lambda *a,**kw: response)
    with pytest.raises(llm.LLMError, match='token budget exhausted'):
        list(llm.generate_stream('s','q'))


def test_vision_does_not_inherit_answer_thinking_or_return_private_content(monkeypatch):
    monkeypatch.setattr(settings, 'llm_answer_thinking', 'enabled')
    send = post(monkeypatch, content='<think>PRIVATE</think>Figure caption.', reasoning='PRIVATE')
    assert llm.generate_vision('s','q',b'image') == 'Figure caption.'
    assert 'reasoning' not in send.call_args.kwargs['json']
    post(monkeypatch, content=None, reasoning='PRIVATE')
    with pytest.raises(llm.LLMError, match='no final answer'):
        llm.generate_vision('s','q',b'image')


def test_budget_error_propagates_without_synthesis_abstention_or_retry(monkeypatch):
    from src.agent import synthesizer
    monkeypatch.setattr(synthesizer, 'build_evidence_pack', lambda state: synthesizer.EvidencePack(context='evidence'))
    generate = MagicMock(side_effect=llm.LLMError('LLM output token budget exhausted'))
    monkeypatch.setattr(synthesizer, 'generate', generate)
    with pytest.raises(llm.LLMError, match='token budget exhausted'):
        synthesizer.synthesize_answer({'question':'q'})
    assert generate.call_count == 1


def test_malformed_provider_response_does_not_echo_private_fields(monkeypatch):
    response = SimpleNamespace(raise_for_status=lambda: None, text='PRIVATE reasoning',
        json=MagicMock(side_effect=json.JSONDecodeError('bad', 'PRIVATE reasoning', 0)))
    monkeypatch.setattr(llm, 'post_json', lambda *a,**kw: response)
    with pytest.raises(llm.LLMError) as caught:
        llm.generate('s','q')
    assert 'PRIVATE' not in str(caught.value)


@pytest.mark.parametrize('mode', ['default', 'enabled', 'disabled'])
@pytest.mark.parametrize('temperature', [0.0, 1.0, 2.0])
def test_answer_controls_reach_provider_payload(monkeypatch, mode, temperature):
    metadata(monkeypatch)
    send = post(monkeypatch)
    monkeypatch.setattr(settings, 'llm_answer_thinking', mode)
    monkeypatch.setattr(settings, 'llm_answer_temperature', temperature)
    llm.generate('s', 'q', **r.answer_generation_kwargs())
    payload = send.call_args.kwargs['json']
    assert payload['temperature'] == temperature
    if mode == 'default':
        assert 'reasoning' not in payload
    else:
        assert payload['reasoning'] == {'enabled': mode == 'enabled', 'exclude': True}


def test_answer_temperature_preserves_fixed_sampling_model_handling(monkeypatch):
    monkeypatch.setattr(settings, 'vllm_model_name', 'o3')
    monkeypatch.setattr(settings, 'llm_answer_temperature', 0.5)
    send = post(monkeypatch)
    llm.generate('s', 'q', **r.answer_generation_kwargs())
    assert 'temperature' not in send.call_args.kwargs['json']
