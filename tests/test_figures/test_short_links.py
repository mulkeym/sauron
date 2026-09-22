import asyncio
import hashlib
import re
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI

from src.config import settings
from src.db.metadata import MetadataStore
from src.figures import links, service
from src.auth.http import EndpointAuthenticationMiddleware
from tests.test_figures.test_figures import populated, figures_root, png
from tests.test_figures.test_links import link_settings, reference, token_of


@pytest.mark.asyncio
async def test_short_links_are_opaque_persistent_and_keep_the_signed_snapshot(tmp_path, figures_root, link_settings):
    ms, record = await populated(tmp_path)
    second = MetadataStore(str(ms.engine.url))
    try:
        doc, ref = await reference(ms, record)
        issued = await links.issue_short(doc, ref, ms)
        token = token_of(issued['inline_url'])
        assert re.fullmatch(links.SHORT_TOKEN_PATTERN, token)
        assert len(token) == 24 and len(issued['inline_url']) < 110
        assert issued['expires_at'] == 1900
        assert doc.doc_id not in token and record.figure_id not in token
        saved = await ms.get_figure_link(hashlib.sha256(token.encode()).hexdigest())
        assert saved is not None and token not in saved.signed_capability
        assert links.verify(saved.signed_capability)['sha'] == ref['sha256']
        # A new store/connection, as used after a process restart, resolves it.
        assert await links.read(token, second) == png()
        legacy = token_of(links.issue(doc, ref)['inline_url'])
        assert await links.read(legacy, second) == png()
    finally:
        await second.engine.dispose()
        await ms.engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['tampered', 'expired', 'rotation', 'acl', 'revision', 'deleted', 'asset', 'disabled'])
async def test_short_link_revocation(tmp_path, figures_root, link_settings, monkeypatch, change):
    ms, record = await populated(tmp_path)
    try:
        doc, ref = await reference(ms, record)
        token = token_of((await links.issue_short(doc, ref, ms))['inline_url'])
        assert await links.read(token, ms) == png()
        if change == 'tampered': token = token[:-1] + ('A' if token[-1] != 'A' else 'B')
        elif change == 'expired': link_settings.now = 1900
        elif change == 'rotation': monkeypatch.setattr(settings, 'figure_link_signing_secret', 'rotated-secret-that-is-at-least-32-chars')
        elif change == 'acl': await ms.update_document('doc-1', acl_groups=['finance'])
        elif change == 'revision': await ms.update_document('doc-1', content_hash='new-revision')
        elif change == 'deleted': await ms.delete_document('doc-1')
        elif change == 'asset': (figures_root / 'doc-1' / record.assets['preview']['key']).write_bytes(b'not an image')
        else: monkeypatch.setattr(settings, 'figure_public_base_url', '')
        with pytest.raises((OSError, ValueError)):
            await links.read(token, ms)
    finally:
        await ms.engine.dispose()


@pytest.mark.asyncio
async def test_issue_cleans_expired_links_and_concurrent_issuance_is_unique(tmp_path, figures_root, link_settings):
    ms, record = await populated(tmp_path)
    try:
        doc, ref = await reference(ms, record)
        first = token_of((await links.issue_short(doc, ref, ms))['inline_url'])
        link_settings.now = 1901
        issued = await asyncio.gather(*(links.issue_short(doc, ref, ms) for _ in range(8)))
        assert len({item['inline_url'] for item in issued}) == 8
        assert await ms.get_figure_link(hashlib.sha256(first.encode()).hexdigest()) is None
        for item in issued:
            assert await links.read(token_of(item['inline_url']), ms) == png()
    finally:
        await ms.engine.dispose()


@pytest.mark.asyncio
async def test_short_route_rejects_bad_tokens_without_requesting_an_api_key(tmp_path, figures_root, link_settings, monkeypatch):
    from src.api import routes_figures
    ms, record = await populated(tmp_path)
    monkeypatch.setattr(routes_figures, 'get_metadata_store', lambda: ms)
    app = FastAPI(root_path='/sauron')
    app.include_router(routes_figures.router)
    app.add_middleware(EndpointAuthenticationMiddleware)
    try:
        doc, ref = await reference(ms, record)
        url = (await links.issue_short(doc, ref, ms))['inline_url']
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='https://diagrams.example.test') as client:
            response = await client.get(url)
            assert response.status_code == 200 and response.content == png()
            assert 'no-store' in response.headers['cache-control']
            assert (await client.head(url)).status_code == 200
            for invalid in ('s_bad', 'invalid', 'x' * 500 + '.' + 'a' * 63 + 's', 's_' + 'X' * 22):
                response = await client.get('/sauron/api/v1/figure-links/' + invalid)
                assert response.status_code == 404
                assert response.json()['detail'] == 'Diagram link is unavailable or expired'
            for method, path in [('POST', url), ('GET', url + '/extra'), ('GET', '/sauron' + ref['content_url']), ('GET', '/sauron/api/health')]:
                assert (await client.request(method, path)).status_code == 403
    finally:
        await ms.engine.dispose()


@pytest.mark.asyncio
async def test_mcp_exact_markdown_catalog_precedes_evidence_and_only_uses_issued_images(tmp_path, figures_root, link_settings):
    ms, record = await populated(tmp_path)
    try:
        _, ref = await reference(ms, record)
        payload = {'answer_with_evidence_ids': 'Branch layout. [Efigure]', 'answer': 'Display answer',
                   'images': [ref], 'citations': [{'evidence_id': 'Efigure', 'filename': 'topology.pdf', 'figure_id': record.figure_id}]}
        result = await service.mcp_result(payload, ['network'], ms)
        body = result.structured_content
        assert list(body)[:2] == ['display_instructions', 'diagram_markdown']
        image = body['images'][0]
        assert re.fullmatch(links.SHORT_TOKEN_PATTERN, token_of(image['inline_url']))
        assert body['diagram_markdown'] == image['markdown']
        assert image['markdown'] in body['answer']
        assert 'Only embed the diagrams in the images list' in body['display_instructions']
        assert await links.read(token_of(image['inline_url']), ms) == png()
        assert 'inline_url' not in ref and 'markdown' not in ref
    finally:
        await ms.engine.dispose()


@pytest.mark.asyncio
async def test_mcp_diagram_search_returns_each_figures_own_ready_to_display_link(tmp_path, figures_root, link_settings, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from src.mcp import server as server_module, activity_wrap
    from src.mcp.auth import MCPContext
    from src.mcp.agent_registry import AgentRegistry
    from src.db.schema_registry import SchemaRegistry
    from src.audit import activity
    ms, record = await populated(tmp_path)
    try:
        doc, first = await service.authorized_figure('doc-1', record.figure_id, ['network'], ms)
        second = {**first, 'figure_id': 'p2-fig-001', 'caption': 'Second topology', 'page': 1}
        await ms.put_figures('doc-1', [first, second])
        candidates = [service.reference(doc, item) for item in (first, second)]
        monkeypatch.setattr(service, 'search_diagrams', AsyncMock(return_value=candidates))
        context = MCPContext(username='reader', groups=['network'], api_key='test-key-1')
        monkeypatch.setattr(server_module, 'current_mcp_context', lambda: context)
        monkeypatch.setattr(activity_wrap, 'current_mcp_context', lambda: context)
        monkeypatch.setattr(activity, 'record_query_activity', AsyncMock())
        server = server_module.create_mcp_server(MagicMock(), SchemaRegistry(), ms, AgentRegistry())
        result = await server.call_tool('tool_search_diagrams', {'query': 'Show both topologies'})
        body = result.structured_content
        refs = body if isinstance(body, list) else body['result']
        assert len(refs) == 2 and refs[0]['inline_url'] != refs[1]['inline_url']
        for ref in refs:
            assert list(ref)[0] == 'markdown'
            assert ref['caption'] in ref['markdown'] and ref['inline_url'] in ref['markdown']
            token = token_of(ref['inline_url'])
            saved = await ms.get_figure_link(hashlib.sha256(token.encode()).hexdigest())
            assert links.verify(saved.signed_capability)['figure'] == ref['figure_id']
            assert await links.read(token, ms) == png()
    finally:
        await ms.engine.dispose()
