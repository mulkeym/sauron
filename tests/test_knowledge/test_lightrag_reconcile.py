"""LightRAG orphan planning + purge helpers (crash / stale doc recovery)."""
from src.knowledge.graph_rag import plan_orphan_lightrag_docs


def test_plan_orphan_drops_docs_not_in_metadata():
    live = {"doc-a", "doc-b"}
    lightrag = {"doc-a", "stale-pdf", "another-orphan"}
    assert plan_orphan_lightrag_docs(live, lightrag) == {"stale-pdf", "another-orphan"}


def test_plan_orphan_empty_metadata_drops_all():
    assert plan_orphan_lightrag_docs(set(), {"x", "y"}) == {"x", "y"}


def test_plan_orphan_keeps_exact_live_set():
    live = {"a", "b", "c"}
    assert plan_orphan_lightrag_docs(live, {"a", "b", "c"}) == set()


def test_plan_orphan_ignores_empty_ids():
    assert plan_orphan_lightrag_docs({""}, {"", "real"}) == {"real"}


import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from lightrag.base import DocStatus
from src.knowledge import graph_rag


@pytest.mark.asyncio
async def test_listing_uses_bulk_status_api_including_every_stage(monkeypatch):
    # Deliberately no get_docs_by_status: LightRAG 1.5.7 removed it.
    storage = SimpleNamespace(get_docs_by_statuses=AsyncMock(return_value={
        'doc-a': SimpleNamespace(status=DocStatus.PROCESSED),
        'doc-b': {'status': 'failed'},
    }))
    rag = SimpleNamespace(doc_status=storage)
    monkeypatch.setattr(graph_rag, '_rag_instance', rag)
    monkeypatch.setattr(graph_rag, 'get_lightrag', AsyncMock(return_value=rag))
    assert await graph_rag.list_lightrag_doc_ids() == {'doc-a': 'processed', 'doc-b': 'failed'}
    storage.get_docs_by_statuses.assert_awaited_once_with(list(DocStatus))


@pytest.mark.asyncio
async def test_failed_status_read_does_not_look_like_an_empty_corpus(monkeypatch):
    rag = SimpleNamespace(doc_status=SimpleNamespace(
        get_docs_by_statuses=AsyncMock(side_effect=OSError('status unreadable')),
    ))
    monkeypatch.setattr(graph_rag, '_rag_instance', rag)
    monkeypatch.setattr(graph_rag, 'get_lightrag', AsyncMock(return_value=rag))
    with pytest.raises(OSError, match='status unreadable'):
        await graph_rag.list_lightrag_doc_ids()


@pytest.mark.asyncio
async def test_insert_cleanup_uses_bulk_status_api(monkeypatch):
    storage = SimpleNamespace(
        get_docs_by_statuses=AsyncMock(return_value={
            'live': SimpleNamespace(status=DocStatus.PROCESSED),
            'stale': SimpleNamespace(status=DocStatus.FAILED),
        }),
        get_by_id=AsyncMock(side_effect=lambda key: None if key == 'stale' else {'status': 'processed'}),
    )
    rag = SimpleNamespace(doc_status=storage, ainsert=AsyncMock(return_value='track'),
        adelete_by_doc_id=AsyncMock(return_value=SimpleNamespace(status="success")),
        chunk_entity_relation_graph=SimpleNamespace(index_done_callback=AsyncMock()))
    store = SimpleNamespace(engine=SimpleNamespace(dispose=AsyncMock()), init=AsyncMock(), list_documents=AsyncMock(
        return_value=[SimpleNamespace(doc_id='live')]))
    monkeypatch.setattr('src.db.metadata.MetadataStore', lambda: store)
    monkeypatch.setattr(graph_rag, 'get_lightrag', AsyncMock(return_value=rag))
    monkeypatch.setattr(graph_rag, '_insert_lock', asyncio.Lock())
    monkeypatch.setattr(graph_rag, '_invalidate_query_cache', lambda: None)
    await graph_rag.insert_document('A connects to B', 'new', 'test.txt')
    storage.get_docs_by_statuses.assert_awaited_once_with(list(DocStatus))
    rag.adelete_by_doc_id.assert_awaited_once_with('stale', delete_llm_cache=True)
    rag.ainsert.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['fail', 'not_allowed', None])
async def test_returned_deletion_failure_is_not_success(status):
    rag = SimpleNamespace(adelete_by_doc_id=AsyncMock(return_value=SimpleNamespace(status=status)),
                          doc_status=SimpleNamespace(get_by_id=AsyncMock(return_value=None)))
    assert not await graph_rag._delete_lightrag_doc(rag, 'old')


@pytest.mark.asyncio
async def test_delete_must_remove_the_status_record():
    rag = SimpleNamespace(adelete_by_doc_id=AsyncMock(return_value=SimpleNamespace(status='success')),
                          doc_status=SimpleNamespace(get_by_id=AsyncMock(return_value={'status': 'failed'})))
    assert not await graph_rag._delete_lightrag_doc(rag, 'old')


@pytest.mark.asyncio
async def test_failed_orphan_cleanup_blocks_insert(monkeypatch):
    rag = SimpleNamespace(doc_status=SimpleNamespace(
        get_docs_by_statuses=AsyncMock(return_value={'stale': {'status': 'failed'}})),
        ainsert=AsyncMock())
    store = SimpleNamespace(init=AsyncMock(), engine=SimpleNamespace(dispose=AsyncMock()),
                            list_documents=AsyncMock(return_value=[]))
    monkeypatch.setattr('src.db.metadata.MetadataStore', lambda: store)
    monkeypatch.setattr(graph_rag, 'get_lightrag', AsyncMock(return_value=rag))
    monkeypatch.setattr(graph_rag, '_delete_lightrag_doc', AsyncMock(return_value=False))
    monkeypatch.setattr(graph_rag, '_insert_lock', asyncio.Lock())
    with pytest.raises(RuntimeError, match='cleanup failed'):
        await graph_rag.insert_document('routers', 'new', 'test.txt')
    rag.ainsert.assert_not_awaited()
