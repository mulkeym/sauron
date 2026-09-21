"""Exercise Sauron's adapter against installed LightRAG with real local stores.

Only the model output, embeddings, and tokenizer are synthetic; no network,
model downloads, or live document data are used.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from src.knowledge.resilient_lightrag import ResilientLightRAG as LightRAG
from lightrag.utils import Tokenizer

from src.knowledge import graph_rag


class CharacterTokenizer:
    def encode(self, text):
        return list(text.encode('utf-8'))

    def decode(self, tokens):
        return bytes(tokens).decode('utf-8', errors='replace')


@pytest.mark.asyncio
@pytest.mark.parametrize('model_mode', ['valid', 'failure', 'no_records', 'delete_reingest', 'retry_failed', 'rebuild_empty', 'rebuild_duplicate', 'delete_empty'])
async def test_real_lightrag_storage_and_extraction(tmp_path, monkeypatch, model_mode):
    workspace = 'test-' + uuid.uuid4().hex
    def factory(**kwargs):
        kwargs.update(working_dir=str(tmp_path), workspace=workspace,
                      tokenizer=Tokenizer('test', CharacterTokenizer()),
                      entity_extract_max_gleaning=0, enable_llm_cache=True)
        return LightRAG(**kwargs)

    async def embed(texts, **kwargs):
        return np.tile(np.array([0.3, 0.4, 0.5], dtype=np.float32), (len(texts), 1))

    output = (
        'entity<|#|>Router Alpha<|#|>Artifact<|#|>Router Alpha is an edge router.\n'
        'entity<|#|>Router Beta<|#|>Artifact<|#|>Router Beta is an edge router.\n'
        'relation<|#|>Router Alpha<|#|>Router Beta<|#|>connects<|#|>Router Alpha connects to Router Beta.\n'
        '<|COMPLETE|>'
    )
    if model_mode in ('failure', 'retry_failed', 'rebuild_duplicate'):
        model = AsyncMock(side_effect=RuntimeError('synthetic model failure'))
    else:
        model = AsyncMock(return_value=output if model_mode in ('valid', 'delete_reingest') else 'The two routers are connected.')
    monkeypatch.setattr(graph_rag, 'LightRAG', factory)
    monkeypatch.setattr(graph_rag, '_detect_embed_dim', lambda: 3)
    monkeypatch.setattr(graph_rag, '_embed_func', embed)
    monkeypatch.setattr(graph_rag, 'openai_complete_if_cache', model)
    monkeypatch.setattr(graph_rag, '_rag_instance', None)
    monkeypatch.setattr(graph_rag, '_initialized', False)
    monkeypatch.setattr(graph_rag, '_insert_lock', asyncio.Lock())
    monkeypatch.setattr(graph_rag, '_invalidate_query_cache', lambda: None)
    monkeypatch.setattr(graph_rag.settings, 'llm_concurrency', 1)
    store = SimpleNamespace(engine=SimpleNamespace(dispose=AsyncMock()), init=AsyncMock(), list_documents=AsyncMock(return_value=[]))
    monkeypatch.setattr('src.db.metadata.MetadataStore', lambda: store)
    rag = await graph_rag.get_lightrag()
    try:
        if model_mode in ('failure', 'retry_failed', 'rebuild_duplicate'):
            with pytest.raises(RuntimeError, match='synthetic model failure'):
                await asyncio.wait_for(graph_rag.insert_document(
                    'Router Alpha connects to Router Beta.', 'doc1', 'synthetic.txt'), timeout=30)
            assert (await graph_rag.list_lightrag_doc_ids())['doc1'] == 'failed'
        else:
            await asyncio.wait_for(graph_rag.insert_document(
                'Router Alpha connects to Router Beta.', 'doc1', 'synthetic.txt'), timeout=30)
            expected = (2, 1) if model_mode in ('valid', 'delete_reingest') else (0, 0)
            assert await graph_rag.get_graph_counts() == expected
            assert (await graph_rag.list_lightrag_doc_ids())['doc1'] == 'processed'
            assert await graph_rag.get_document_graph_counts('doc1') == expected
            saved = next(tmp_path.rglob('graph_chunk_entity_relation.graphml'))
            xml = ET.parse(saved)
            assert len(xml.findall('.//{*}node')) == expected[0]
            assert len(xml.findall('.//{*}edge')) == expected[1]
        if model_mode in ('delete_reingest', 'delete_empty'):
            model.return_value = output
            calls_before = model.await_count
            # Keep another catalog ID so this exercises per-document cleanup,
            # not the empty-corpus hard purge shortcut.
            await graph_rag.reconcile_lightrag_with_metadata({'other'})
            assert await rag.doc_status.get_by_id('doc1') is None
            await graph_rag.insert_document('Router Alpha connects to Router Beta.', 'doc2', 'synthetic.txt')
            assert (await rag.doc_status.get_by_id('doc2'))['status'] == 'processed'
            assert model.await_count > calls_before
            assert await graph_rag.get_graph_counts() == (2, 1)
        elif model_mode == 'rebuild_duplicate':
            # Reproduce the broken-version state: metadata was deleted but the
            # failed LightRAG primary remained; a second upload made a dup row.
            await rag.ainsert('Router Alpha connects to Router Beta.', ids=['doc2'], file_paths=['synthetic.txt'])
            assert any(key.startswith('dup-') for key in await graph_rag.list_lightrag_doc_ids())
            store.list_documents.return_value = [SimpleNamespace(doc_id='doc2')]
            model.side_effect = None
            model.return_value = output
            await graph_rag.insert_document('Router Alpha connects to Router Beta.', 'doc2', 'synthetic.txt', rebuild=True)
            assert await graph_rag.list_lightrag_doc_ids() == {'doc2': 'processed'}
            assert await graph_rag.get_graph_counts() == (2, 1)
        elif model_mode in ('retry_failed', 'rebuild_empty'):
            model.side_effect = None
            model.return_value = output
            store.list_documents.return_value = [SimpleNamespace(doc_id='doc1')]
            await graph_rag.insert_document('Router Alpha connects to Router Beta.', 'doc1', 'synthetic.txt',
                                           rebuild=model_mode == 'rebuild_empty')
            assert (await rag.doc_status.get_by_id('doc1'))['status'] == 'processed'
            assert await graph_rag.get_graph_counts() == (2, 1)
        assert model.await_count >= 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 3])
async def test_middle_chunk_failure_keeps_later_graph_and_retry_uses_cache(tmp_path, monkeypatch, concurrency):
    from src.knowledge.resilient_lightrag import FAILURES_KEY
    workspace = 'partial-' + uuid.uuid4().hex
    def factory(**kwargs):
        kwargs.update(working_dir=str(tmp_path), workspace=workspace,
                      tokenizer=Tokenizer('test', CharacterTokenizer()),
                      entity_extract_max_gleaning=0, enable_llm_cache=True)
        return LightRAG(**kwargs)
    async def embed(texts, **kwargs):
        return np.tile(np.array([0.3, 0.4, 0.5], dtype=np.float32), (len(texts), 1))
    failed = True
    successful_prompts = []
    async def model(**kwargs):
        prompt = kwargs['prompt']
        if 'FAIL_MIDDLE' in prompt and failed:
            raise RuntimeError('synthetic middle chunk timeout')
        successful_prompts.append(prompt)
        name = 'Last Router' if 'OK_LAST' in prompt else 'First Router'
        return f'entity<|#|>{name}<|#|>Artifact<|#|>{name} is an edge router.\n<|COMPLETE|>'
    monkeypatch.setattr(graph_rag, 'LightRAG', factory)
    monkeypatch.setattr(graph_rag, '_detect_embed_dim', lambda: 3)
    monkeypatch.setattr(graph_rag, '_embed_func', embed)
    monkeypatch.setattr(graph_rag, 'openai_complete_if_cache', model)
    monkeypatch.setattr(graph_rag, '_rag_instance', None)
    monkeypatch.setattr(graph_rag, '_initialized', False)
    monkeypatch.setattr(graph_rag, '_insert_lock', asyncio.Lock())
    monkeypatch.setattr(graph_rag, '_invalidate_query_cache', lambda: None)
    monkeypatch.setattr(graph_rag.settings, 'llm_concurrency', concurrency)
    monkeypatch.setattr(graph_rag.settings, 'kg_chunk_token_size', 200)
    monkeypatch.setattr(graph_rag.settings, 'kg_chunk_overlap_token_size', 0)
    store = SimpleNamespace(engine=SimpleNamespace(dispose=AsyncMock()), init=AsyncMock(),
                            list_documents=AsyncMock(return_value=[SimpleNamespace(doc_id='doc1')]))
    monkeypatch.setattr('src.db.metadata.MetadataStore', lambda: store)
    text = 'OK_FIRST ' + 'a' * 191 + 'FAIL_MIDDLE ' + 'b' * 188 + 'OK_LAST ' + 'c' * 190
    rag = await graph_rag.get_lightrag()
    try:
        with pytest.raises(RuntimeError, match='incomplete'):
            await graph_rag.insert_document(text, 'doc1', 'partial.txt')
        row = await rag.doc_status.get_by_id('doc1')
        assert row['status'] == 'failed'
        assert len(row['metadata'][FAILURES_KEY]) == 1
        assert any('OK_LAST' in p for p in successful_prompts), 'later chunks must execute'
        assert (await graph_rag.get_document_graph_counts('doc1'))[0] == 2
        # Verify failure markers survive a disk flush and successful graph is visible.
        status_file = next(tmp_path.rglob('kv_store_doc_status.json'))
        import json
        assert json.loads(status_file.read_text())['doc1']['metadata'][FAILURES_KEY]
        already_successful = set(successful_prompts)
        successful_prompts.clear()
        failed = False
        await graph_rag.insert_document(text, 'doc1', 'partial.txt', rebuild=True)
        assert (await rag.doc_status.get_by_id('doc1'))['status'] == 'processed'
        assert not already_successful.intersection(successful_prompts), 'successful extraction must use cache'
        assert any('FAIL_MIDDLE' in p for p in successful_prompts)
        assert (await graph_rag.get_document_graph_counts('doc1'))[0] == 2
    finally:
        await rag.finalize_storages()
