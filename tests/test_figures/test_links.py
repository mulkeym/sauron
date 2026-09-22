import base64
import json
from types import SimpleNamespace
from urllib.parse import urlsplit
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from src.config import Settings, settings
from src.figures import links, service, presentation
from src.auth.http import EndpointAuthenticationMiddleware
from tests.test_figures.test_figures import populated, figures_root, png


@pytest.fixture
def link_settings(monkeypatch):
    monkeypatch.setattr(settings, "figure_public_base_url", "https://diagrams.example.test/sauron")
    monkeypatch.setattr(settings, "figure_link_signing_secret", "test-diagram-signing-secret-32-characters")
    monkeypatch.setattr(settings, "figure_link_ttl_seconds", 900)
    clock = SimpleNamespace(now=1000)
    monkeypatch.setattr(links, "time", SimpleNamespace(time=lambda: clock.now))
    return clock


async def reference(ms, record):
    doc, figure = await service.authorized_figure("doc-1", record.figure_id, ["network"], ms)
    ref = service.reference(doc, figure)
    return doc, {**ref, "evidence_id": "Efigure", "evidence_ids": ["Efigure"]}


def token_of(url):
    return url.rsplit("/", 1)[-1]


@pytest.mark.asyncio
async def test_png_capability_tamper_expiry_and_key_rotation(tmp_path, figures_root, link_settings, monkeypatch):
    ms, record = await populated(tmp_path)
    try:
        doc, ref = await reference(ms, record)
        issued = links.issue(doc, ref)
        assert issued["inline_url"].startswith("https://diagrams.example.test/sauron/api/v1/figure-links/")
        token = token_of(issued["inline_url"])
        assert await links.read(token, ms) == png()
        encoded, signature = token.split(".")
        data = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        assert data["sha"] == ref["sha256"] and data["exp"] == 1900
        assert "network" not in json.dumps(data)
        for key, value in [("exp", 99999), ("doc", "other-doc"), ("figure", "other-figure"), ("variant", "full"), ("sha", "0"*64)]:
            modified = base64.urlsafe_b64encode(json.dumps({**data, key:value}).encode()).rstrip(b"=").decode()
            with pytest.raises(ValueError):
                await links.read(modified + "." + signature, ms)
        link_settings.now = 1899
        assert await links.read(token, ms) == png()
        link_settings.now = 1900
        with pytest.raises(ValueError):
            await links.read(token, ms)
        link_settings.now = 1000
        monkeypatch.setattr(settings, "figure_link_signing_secret", "rotated-secret-that-is-at-least-32-chars")
        with pytest.raises(ValueError):
            await links.read(token, ms)
    finally:
        await ms.engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["acl", "revision", "deleted", "asset"])
async def test_live_document_and_asset_changes_revoke_links(tmp_path, figures_root, link_settings, change):
    ms, record = await populated(tmp_path)
    try:
        doc, ref = await reference(ms, record)
        token = token_of(links.issue(doc, ref)["inline_url"])
        if change == "acl":
            await ms.update_document("doc-1", acl_groups=["finance"])
        elif change == "revision":
            await ms.update_document("doc-1", content_hash="replacement-source")
        elif change == "deleted":
            await ms.delete_document("doc-1")
        else:
            (figures_root / "doc-1" / record.assets["preview"]["key"]).write_bytes(b"not a PNG")
        with pytest.raises((OSError, ValueError)):
            await links.read(token, ms)
    finally:
        await ms.engine.dispose()


@pytest.mark.asyncio
async def test_only_exact_signed_read_bypasses_api_key_gate(tmp_path, figures_root, link_settings, monkeypatch):
    from src.api import routes_figures
    ms, record = await populated(tmp_path)
    monkeypatch.setattr(routes_figures, "get_metadata_store", lambda: ms)
    app = FastAPI(root_path="/sauron")
    app.include_router(routes_figures.router)
    app.add_middleware(EndpointAuthenticationMiddleware)
    try:
        doc, ref = await reference(ms, record)
        url = links.issue(doc, ref)["inline_url"]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://diagrams.example.test") as client:
            response = await client.get(url)
            assert response.status_code == 200 and response.content == png()
            assert response.headers["content-type"] == "image/png"
            assert "no-store" in response.headers["cache-control"]
            assert response.headers["x-content-type-options"] == "nosniff"
            head = await client.head(url)
            assert head.status_code == 200 and not head.content
            assert (await client.post(url)).status_code == 403
            assert (await client.get(url + "/extra")).status_code == 403
            assert (await client.get("/sauron" + ref["content_url"])).status_code == 403
            forged = url[:-1] + ("0" if url[-1] != "0" else "1")
            assert (await client.get(forged)).status_code == 404
            link_settings.now = 1900
            response = await client.get(url)
            assert response.status_code == 404 and "no-store" in response.headers["cache-control"]
    finally:
        await ms.engine.dispose()


@pytest.mark.asyncio
async def test_delivery_is_authorized_fresh_and_does_not_mutate_cached_answer(tmp_path, figures_root, link_settings, monkeypatch):
    ms, record = await populated(tmp_path)
    canonical = "SD-WAN connects these sites. [Efigure]\n\nOther considerations."
    try:
        _, ref = await reference(ms, record)
        answer1, refs1 = await presentation.present_answer(canonical, [ref], ["network"], ms)
        link_settings.now += 950
        answer2, refs2 = await presentation.present_answer(canonical, [ref], ["network"], ms)
        assert refs1[0]["inline_url"] != refs2[0]["inline_url"]
        assert answer2.index("![") < answer2.index("Other considerations")
        assert "inline_url" not in ref and "![" not in canonical
        with pytest.raises(ValueError):
            await links.read(token_of(refs1[0]["inline_url"]), ms)
        assert await links.read(token_of(refs2[0]["inline_url"]), ms) == png()
        unauthorized, empty = await presentation.present_answer(canonical, [ref], ["finance"], ms)
        assert unauthorized == canonical and empty == []
        stale, empty = await presentation.present_answer(canonical, [{**ref, "sha256":"bad-hash"}], ["network"], ms)
        assert stale == canonical and empty == []
        monkeypatch.setattr(settings, "figure_public_base_url", "")
        plain, refs = await presentation.present_answer(canonical, [ref], ["network"], ms)
        assert plain == canonical and "inline_url" not in refs[0]
        with pytest.raises(ValueError):
            await links.read(token_of(refs2[0]["inline_url"]), ms)
    finally:
        await ms.engine.dispose()


@pytest.mark.asyncio
async def test_mcp_links_are_inline_without_duplicate_native_images(tmp_path, figures_root, link_settings):
    ms, record = await populated(tmp_path)
    try:
        _, ref = await reference(ms, record)
        canonical = "Explain the branch layout. [Efigure]\n\nNext topic."
        payload = {"answer_with_evidence_ids": canonical, "answer":"display answer", "images":[ref],
                   "citations":[{"evidence_id":"Efigure", "filename":"topology.pdf", "figure_id":record.figure_id}]}
        result = await service.mcp_result(payload, ["network"], ms)
        assert [block.type for block in result.content] == ["text"]
        body = result.structured_content
        assert body["answer"].index("![") < body["answer"].index("Next topic")
        assert body["images"][0]["expires_at"] == 1900
        assert "display_instructions" in body
        assert payload["answer_with_evidence_ids"] == canonical
        explicit = await service.mcp_result({"images":[ref]}, ["network"], ms)
        assert explicit.structured_content["diagram_markdown"].startswith("![")
    finally:
        await ms.engine.dispose()


@pytest.mark.parametrize("url", ["/sauron", "ftp://example.com", "https://user:pass@example.com", "https://example.com?q=1",
    "https://example.com/#fragment", "https://example.com/a)b", "https://example.com/\nattack", "https://example.com:bad"])
def test_public_url_validation(url):
    with pytest.raises(ValueError):
        Settings(figure_public_base_url=url)


def test_link_settings_validation_and_secret_redaction(link_settings):
    from src.admin.settings_catalog import settings_catalog, prepare_update
    assert Settings(figure_public_base_url="https://example.com/proxy/").figure_public_base_url == "https://example.com/proxy"
    with pytest.raises(ValueError):
        Settings(figure_link_signing_secret="short")
    with pytest.raises(ValueError):
        Settings(figure_link_ttl_seconds=0)
    fields = {item['name']:item for items in settings_catalog().values() for item in items}
    assert fields['figure_link_signing_secret']['secret']
    assert fields['figure_link_signing_secret']['value'] == ''
    assert prepare_update({'figure_link_signing_secret':'', 'keep_blank_secrets':'true'})['figure_link_signing_secret'] == settings.figure_link_signing_secret


def test_inline_placement_skips_code_links_and_deduplicates():
    refs = [{"doc_id":"d", "figure_id":"f", "evidence_id":"Efigure", "caption":"Branch [untrusted] <img>",
             "inline_url":"https://example.test/image", "evidence_ids":["Efigure", "Eother"]}]
    answer = "```text\n[Efigure]\n```\n\n`[Efigure]` and [Efigure](https://example.test).\n\nActual discussion. [Eother]\n\nLater. [Efigure]"
    rendered = presentation.inline_images(answer, refs + refs)
    assert rendered.count("![") == 1
    assert rendered.index("Actual discussion") < rendered.index("![") < rendered.index("Later.")
    assert "<img>" not in rendered and r"\[untrusted\]" in rendered
    assert presentation.inline_images("No figure citation. [Eunknown]", refs) == "No figure citation. [Eunknown]"


@pytest.mark.asyncio
@pytest.mark.parametrize('endpoint', ['/v1/chat/completions', '/api/v1/query'])
async def test_real_chat_delivery_contains_inline_fetchable_png(tmp_path, figures_root, link_settings, monkeypatch, endpoint):
    from src.api import routes_openai_compat, routes_query, routes_figures
    from src.auth.jwt import create_token
    from src.audit import activity
    from src.generation.rag_chain import RAGResponse
    from src.retrieval.models import Citation
    from src.main import create_app
    ms, record = await populated(tmp_path)
    try:
        _, ref = await reference(ms, record)
        canonical = 'Government SD-WAN connects these sites. [Efigure]\n\nOperational considerations.'
        citation = Citation(doc_id='doc-1', filename='topology.pdf', doc_type='pdf', chunk_index=0,
            snippet='Source diagram', relevance=.9, figure_id=record.figure_id, evidence_id='Efigure', page=1)
        response = RAGResponse(answer=canonical, citations=[citation], images=[ref])
        for module in (routes_openai_compat, routes_query):
            monkeypatch.setattr(module, 'agent_query', AsyncMock(return_value=response))
            monkeypatch.setattr(module, 'get_metadata_store', lambda: ms)
            monkeypatch.setattr(module, 'get_vector_store', lambda: None)
            monkeypatch.setattr(module, 'get_schema_registry', lambda: None)
        monkeypatch.setattr(routes_figures, 'get_metadata_store', lambda: ms)
        monkeypatch.setattr(activity, 'record_query_activity', AsyncMock())
        monkeypatch.setattr(settings, 'figure_public_base_url', 'https://sauron.test')
        jwt = create_token('reader', ['network'])
        headers = {'X-API-Key':'test-key-1', 'Authorization':f'Bearer {jwt}'}
        body = ({'model':'sauron', 'messages':[{'role':'user','content':'Explain SD-WAN for government to me'}]}
                if endpoint.startswith('/v1') else {'question':'Explain SD-WAN for government to me'})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url='https://sauron.test') as client:
            delivered = await client.post(endpoint, headers=headers, json=body)
            assert delivered.status_code == 200, delivered.text
            payload = delivered.json()
            answer = payload['choices'][0]['message']['content'] if endpoint.startswith('/v1') else payload['answer']
            refs = payload['sauron_images'] if endpoint.startswith('/v1') else payload['images']
            assert answer.index('![') < answer.index('Operational considerations')
            assert refs[0]['inline_url'] in answer
            public = await client.get(refs[0]['inline_url'])  # No application key or user JWT.
            assert public.status_code == 200 and public.content == png()
            assert response.answer == canonical  # Cache / jobs retain reusable canonical content.
    finally:
        await ms.engine.dispose()
