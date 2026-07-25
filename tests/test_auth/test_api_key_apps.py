"""DB-backed application API keys."""
import pytest
import pytest_asyncio
from src.auth.api_key import (
    generate_api_key,
    hash_api_key,
    key_prefix,
    reload_api_key_cache,
    resolve_api_key,
    set_db_key_cache,
    validate_api_key,
)
from src.db.metadata import MetadataStore


@pytest_asyncio.fixture
async def store(tmp_path):
    db = tmp_path / "test_api_keys.db"
    s = MetadataStore(database_url=f"sqlite+aiosqlite:///{db}")
    # create tables only — skip full init seed against real settings
    async with s.engine.begin() as conn:
        from src.db.models import Base
        await conn.run_sync(Base.metadata.create_all)
    set_db_key_cache({})
    yield s
    set_db_key_cache({})


@pytest.mark.asyncio
async def test_create_app_and_validate_key(store):
    app = await store.add_api_application(
        name="sdwan-demo-chat",
        display_name="SD-WAN Demo",
        description="Demo backend",
    )
    assert app is not None
    assert app.name == "sdwan-demo-chat"

    secret = generate_api_key()
    assert secret.startswith("sk_")
    rec = await store.add_api_key_record(
        application_id=app.id,
        key_hash=hash_api_key(secret),
        key_prefix=key_prefix(secret),
        label="test",
    )
    assert rec.id
    await reload_api_key_cache(store)

    assert validate_api_key(secret) is True
    ctx = resolve_api_key(secret)
    assert ctx is not None
    assert ctx.source == "db"
    assert ctx.application_name == "sdwan-demo-chat"
    assert validate_api_key("totally-bogus") is False


@pytest.mark.asyncio
async def test_revoke_key(store):
    app = await store.add_api_application(name="openwebui", display_name="OpenWebUI")
    secret = generate_api_key()
    rec = await store.add_api_key_record(
        application_id=app.id,
        key_hash=hash_api_key(secret),
        key_prefix=key_prefix(secret),
    )
    await reload_api_key_cache(store)
    assert validate_api_key(secret) is True

    await store.revoke_api_key(rec.id)
    await reload_api_key_cache(store)
    assert validate_api_key(secret) is False


@pytest.mark.asyncio
async def test_deactivate_app_disables_keys(store):
    app = await store.add_api_application(name="mcp-gateway", display_name="MCP")
    secret = generate_api_key()
    await store.add_api_key_record(
        application_id=app.id,
        key_hash=hash_api_key(secret),
        key_prefix=key_prefix(secret),
    )
    await reload_api_key_cache(store)
    assert validate_api_key(secret) is True

    await store.update_api_application(app.id, active=False)
    await reload_api_key_cache(store)
    assert validate_api_key(secret) is False


@pytest.mark.asyncio
async def test_legacy_settings_key_still_works(store):
    # Ensure DB cache empty; settings still has test-key-1 from conftest
    set_db_key_cache({})
    assert validate_api_key("test-key-1") is True
    ctx = resolve_api_key("test-key-1")
    assert ctx is not None
    assert ctx.source == "legacy"
