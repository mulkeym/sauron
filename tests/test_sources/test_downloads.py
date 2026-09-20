import hashlib
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import jwt
import pytest
import pytest_asyncio
from fastapi import FastAPI

from src.api import routes_sources
from src.auth.http import EndpointAuthenticationMiddleware
from src.config import settings
from src.db.metadata import MetadataStore
from src.sources.service import document_download
from src.sources.storage import OriginalStore

SECRET = "download-test-secret-32-bytes-long!"
RAW = b"%PDF-1.7\n" + bytes(range(256)) * 5000
REVISION = hashlib.sha256(RAW).hexdigest()
PATH = f"/api/v1/documents/doc-1/revisions/{REVISION}/original"


@pytest_asyncio.fixture
async def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "source_originals_dir", str(tmp_path / "originals"))
    monkeypatch.setattr(settings, "source_download_jwt_secret", SECRET)
    monkeypatch.setattr(
        settings, "source_download_webui_url", "https://chat.example.test"
    )
    store = MetadataStore("sqlite+aiosqlite:///" + str(tmp_path / "metadata.db"))
    await store.init()
    await store.add_document(
        "doc-1", "diagram.pdf", "pdf", ["engineering"], 1, "test", content_hash=REVISION
    )
    source = tmp_path / "original.pdf"
    source.write_bytes(RAW)
    OriginalStore().retain(source, "doc-1", REVISION)
    source.unlink()
    monkeypatch.setattr(routes_sources, "get_metadata_store", lambda: store)
    app = FastAPI()
    app.include_router(routes_sources.router)
    app.add_middleware(EndpointAuthenticationMiddleware)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://sauron"
    ) as client:
        yield client, store
    await store.engine.dispose()


def headers(**claims):
    now = int(time.time())
    payload = dict(
        sub="alice-id",
        iss="open-webui",
        aud="sauron-source-downloads",
        iat=now,
        exp=now + 60,
        groups=["engineering"],
        doc_id="doc-1",
        revision=REVISION,
        operation="read-original",
    )
    payload.update(claims)
    return {
        "X-API-Key": "test-key-1",
        "X-Sauron-Download-Identity": jwt.encode(payload, SECRET, algorithm="HS256"),
    }


@pytest.mark.asyncio
async def test_authorized_exact_original_and_private_headers(setup, caplog):
    client, _ = setup
    caplog.set_level("INFO", logger="sauron.source_access")
    response = await client.get(PATH, headers=headers())
    assert response.status_code == 200
    assert response.content == RAW
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "alice-id" in caplog.text and REVISION in caplog.text
    assert SECRET not in caplog.text and "test-key-1" not in caplog.text
    assert headers()["X-Sauron-Download-Identity"] not in caplog.text


@pytest.mark.asyncio
async def test_copied_link_and_forged_headers_cannot_grant_access(setup):
    client, _ = setup
    assert (
        await client.get(PATH, headers=headers(sub="bob-id", groups=["sales"]))
    ).status_code == 404
    assert (await client.get(PATH, headers=headers(groups=["ALL"]))).status_code == 404
    assert (
        await client.get(
            PATH,
            headers={
                "X-API-Key": "test-key-1",
                "X-OpenWebUI-User-Name": "alice",
                "X-Sauron-User-Groups": "engineering",
            },
        )
    ).status_code == 401
    assert (await client.get(PATH)).status_code == 403
    forged = headers()
    forged["X-Sauron-Download-Identity"] += "invalid"
    assert (await client.get(PATH, headers=forged)).status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claims",
    [
        {"aud": "other-service"},
        {"sub": ""},
        {"operation": "preview"},
        {"doc_id": "other"},
        {"revision": "a" * 64},
        {"exp": int(time.time()) - 30},
        {"exp": int(time.time()) + 600},
        {"iat": int(time.time()) + 100},
        {"groups": "engineering"},
        {"groups": [123]},
        {"iss": "attacker"},
    ],
)
async def test_assertion_is_short_lived_scoped_and_validated(setup, claims):
    client, _ = setup
    assert (await client.get(PATH, headers=headers(**claims))).status_code == 401


@pytest.mark.asyncio
async def test_revocation_deletion_and_dataset_disable_rechecked(setup):
    client, store = setup
    assert (await client.get(PATH, headers=headers())).status_code == 200
    await store.update_document("doc-1", acl_groups=["sales"])
    assert (await client.get(PATH, headers=headers())).status_code == 404
    await store.update_document("doc-1", acl_groups=["engineering"], dataset_id=99)
    assert (await client.head(PATH, headers=headers())).status_code == 404
    dataset = await store.add_dataset(
        "Engineering", "engineering", default_acl_groups=["sales"]
    )
    await store.update_document("doc-1", dataset_id=dataset.id)
    # Dataset defaults do not override the current document ACL.
    assert (await client.get(PATH, headers=headers())).status_code == 200
    from sqlalchemy import update
    from src.db.models import Dataset

    async with store.session_factory() as session:
        await session.execute(
            update(Dataset).where(Dataset.id == dataset.id).values(active=False)
        )
        await session.commit()
    assert (
        await client.get(PATH, headers={**headers(), "Range": "bytes=0-7"})
    ).status_code == 404
    await store.delete_document("doc-1")
    assert (await client.get(PATH, headers=headers())).status_code == 404
    assert not (OriginalStore().root / "doc-1").exists()


@pytest.mark.asyncio
async def test_revision_never_falls_forward_and_missing_source_is_clear(setup):
    client, store = setup
    await store.update_document("doc-1", content_hash="b" * 64)
    assert (await client.get(PATH, headers=headers())).status_code == 409
    await store.update_document("doc-1", content_hash=REVISION)
    OriginalStore().delete_document("doc-1")
    response = await client.get(PATH, headers=headers())
    assert response.status_code == 410 and "Original unavailable" in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "range_value,start,end",
    [
        ("bytes=3-18", 3, 18),
        ("bytes=-17", len(RAW) - 17, len(RAW) - 1),
        ("bytes=100-", 100, len(RAW) - 1),
    ],
)
async def test_ranges(setup, range_value, start, end):
    client, _ = setup
    response = await client.get(PATH, headers={**headers(), "Range": range_value})
    assert response.status_code == 206 and response.content == RAW[start : end + 1]
    assert response.headers["content-range"] == f"bytes {start}-{end}/{len(RAW)}"
    assert int(response.headers["content-length"]) == end - start + 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        "bytes=1-2,5-6",
        "bytes=-0",
        "bytes=999999999-",
        "bytes=8-3",
        "garbage",
        "bytes=-",
    ],
)
async def test_invalid_ranges(setup, value):
    client, _ = setup
    response = await client.get(PATH, headers={**headers(), "Range": value})
    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{len(RAW)}"


@pytest.mark.asyncio
async def test_head_and_if_range(setup):
    client, _ = setup
    response = await client.head(PATH, headers=headers())
    assert response.status_code == 200 and not response.content
    assert int(response.headers["content-length"]) == len(RAW)
    response = await client.get(
        PATH, headers={**headers(), "Range": "bytes=0-3", "If-Range": '"wrong"'}
    )
    assert response.status_code == 200 and response.content == RAW


@pytest.mark.asyncio
async def test_mcp_result_only_contains_identifiers_and_login_link(setup):
    _, store = setup
    result = await document_download("doc-1", REVISION, ["engineering"], store)
    assert result["available"]
    assert result[
        "download_url"
    ] == "https://chat.example.test/api/v1/sauron" + PATH.removeprefix("/api/v1")
    serialized = json.dumps(result)
    assert SECRET not in serialized and "test-key-1" not in serialized
    assert "?" not in result["download_url"] and "token" not in serialized.lower()
    assert not (await document_download("doc-1", REVISION, ["sales"], store))[
        "available"
    ]


@pytest.mark.asyncio
async def test_corrupt_source_or_symlinks_never_served(setup, tmp_path):
    client, _ = setup
    path = OriginalStore().root / "doc-1" / (REVISION + ".blob")
    path.chmod(0o600)
    path.write_bytes(b"corrupt")
    assert (await client.get(PATH, headers=headers())).status_code == 410
    outside = tmp_path / "outside"
    outside.write_bytes(RAW)
    path.unlink()
    path.symlink_to(outside)
    assert (await client.get(PATH, headers=headers())).status_code == 410
    path.unlink()
    path.parent.rmdir()
    path.parent.symlink_to(tmp_path, target_is_directory=True)
    assert (await client.get(PATH, headers=headers())).status_code == 410


def test_storage_hash_backfill_and_immutable_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "source_originals_dir", str(tmp_path / "store"))
    source = tmp_path / "source"
    source.write_bytes(RAW)
    store = OriginalStore()
    with pytest.raises(ValueError):
        store.retain(source, "../escape", REVISION)
    with pytest.raises(ValueError):
        store.retain(source, "doc-1", "0" * 64)
    store.retain(source, "doc-1", REVISION)
    store.retain(source, "doc-1", REVISION)
    stream, size = store.open_verified("doc-1", REVISION)
    assert stream.read() == RAW and size == len(RAW)
    stream.close()
    assert len(list((store.root / "doc-1").iterdir())) == 1


def test_storage_is_explicitly_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "source_originals_dir", "")
    assert not OriginalStore().retain(tmp_path / "absent", "doc-1", REVISION)


@pytest.mark.asyncio
async def test_disconnect_before_body_still_closes_original():
    import io
    from starlette.requests import ClientDisconnect

    stream = io.BytesIO(RAW)

    async def chunks():
        yield stream.read()

    response = routes_sources.OriginalFileResponse(chunks(), stream=stream)

    async def send(message):
        raise OSError("Client disconnected before headers")

    with pytest.raises(ClientDisconnect):
        await response(
            {"type": "http", "asgi": {"spec_version": "2.4"}}, AsyncMock(), send
        )
    assert stream.closed


@pytest.mark.asyncio
async def test_visio_original_downloads_without_conversion(setup, tmp_path):
    from tests.test_ingestion.test_visio import fixture

    client, store = setup
    source = fixture(tmp_path / "Topology.vsdx")
    raw = source.read_bytes()
    revision = hashlib.sha256(raw).hexdigest()
    await store.add_document(
        "visio-1",
        source.name,
        "vsdx",
        ["engineering"],
        1,
        "test",
        content_hash=revision,
    )
    OriginalStore().retain(source, "visio-1", revision, source.name)
    response = await client.get(
        f"/api/v1/documents/visio-1/revisions/{revision}/original",
        headers=headers(doc_id="visio-1", revision=revision),
    )
    assert response.status_code == 200 and response.content == raw
    assert response.headers["content-type"] == "application/vnd.visio"
    assert response.headers["content-disposition"].startswith("attachment;")


@pytest.mark.asyncio
async def test_emf_original_download_is_exact_attachment(setup, tmp_path):
    from tests.test_ingestion.test_emf import sample_emf
    client, store = setup
    source = tmp_path / 'Topology.emf'; raw = sample_emf(); source.write_bytes(raw)
    revision = hashlib.sha256(raw).hexdigest()
    await store.add_document('emf-1', source.name, 'emf', ['engineering'], 1, 'test', content_hash=revision)
    assert OriginalStore().retain(source, 'emf-1', revision, source.name)
    response = await client.get(f'/api/v1/documents/emf-1/revisions/{revision}/original',
        headers=headers(doc_id='emf-1', revision=revision))
    assert response.status_code == 200 and response.content == raw
    assert response.headers['content-type'] == 'application/octet-stream'
    assert response.headers['content-disposition'].startswith('attachment;')
