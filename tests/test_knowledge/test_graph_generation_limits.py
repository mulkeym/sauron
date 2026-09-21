import asyncio
from unittest.mock import AsyncMock

import pytest
from lightrag.utils import TruncatedResponse
from src.knowledge import graph_rag


@pytest.mark.asyncio
async def test_graph_limits_and_no_thinking_are_independent_of_answers(monkeypatch):
    model = AsyncMock(return_value='<|COMPLETE|>')
    monkeypatch.setattr(graph_rag, 'openai_complete_if_cache', model)
    monkeypatch.setattr(graph_rag.settings, 'vllm_base_url', 'http://model.local/v1')
    monkeypatch.setattr(graph_rag.settings, 'vllm_model_name', 'gemma-4-26b')
    monkeypatch.setattr(graph_rag.settings, 'llm_reasoning_adapter', 'vllm_template')
    monkeypatch.setattr(graph_rag.settings, 'llm_answer_thinking', 'enabled')
    await graph_rag._llm_func('Extract', max_tokens=20000,
                             extra_body={'other': True, 'chat_template_kwargs': {'enable_thinking': True}})
    args = model.call_args.kwargs
    assert args['max_tokens'] == 4096
    assert args['extra_body'] == {'other': True, 'chat_template_kwargs': {'enable_thinking': False}}
    assert args['enable_cot'] is False
    assert args['timeout'] == 180


@pytest.mark.asyncio
async def test_truncated_output_is_not_cached_as_complete(monkeypatch):
    monkeypatch.setattr(graph_rag, 'openai_complete_if_cache', AsyncMock(return_value=TruncatedResponse('entity partial')))
    with pytest.raises(RuntimeError, match='token limit'):
        await graph_rag._llm_func('Extract')


@pytest.mark.asyncio
async def test_deadline_bounds_adapter_including_retries(monkeypatch):
    cancelled = asyncio.Event()
    async def hung(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    monkeypatch.setattr(graph_rag, 'openai_complete_if_cache', hung)
    monkeypatch.setattr(graph_rag.settings, 'kg_llm_timeout_seconds', 0.02)
    with pytest.raises(RuntimeError, match='deadline'):
        await asyncio.wait_for(graph_rag._llm_func('Extract'), 2)
    assert cancelled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['cancel', 'storage'])
async def test_fatal_errors_drain_workers_without_attempting_more_chunks(monkeypatch, failure):
    from lightrag import LightRAG
    from lightrag.exceptions import PipelineCancelledException
    from src.knowledge.resilient_lightrag import ResilientLightRAG
    # No stores needed: exercise scheduling with a controllable upstream method.
    rag = object.__new__(ResilientLightRAG)
    rag.llm_model_max_async = 1
    calls = []
    async def extract(self, chunks, *args, **kwargs):
        calls.extend(chunks)
        if failure == 'cancel':
            raise PipelineCancelledException('cancelled by user')
        raise OSError('disk full')
    monkeypatch.setattr(LightRAG, '_process_extract_entities', extract)
    expected = PipelineCancelledException if failure == 'cancel' else OSError
    with pytest.raises(expected):
        await rag._process_extract_entities({str(i): {'full_doc_id': 'doc'} for i in range(3)})
    assert calls == ['0']


@pytest.mark.asyncio
async def test_external_cancellation_drains_active_chunk(monkeypatch):
    from lightrag import LightRAG
    from src.knowledge.resilient_lightrag import ResilientLightRAG
    rag = object.__new__(ResilientLightRAG)
    rag.llm_model_max_async = 1
    started, ended = asyncio.Event(), asyncio.Event()
    async def extract(self, *args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            ended.set()
    monkeypatch.setattr(LightRAG, '_process_extract_entities', extract)
    task = asyncio.create_task(rag._process_extract_entities({'first': {'full_doc_id': 'doc'}}))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ended.is_set()
