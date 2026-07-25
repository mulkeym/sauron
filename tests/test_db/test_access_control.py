"""Tests for ACL groups, personas, and playground group resolution."""
import pytest
import pytest_asyncio

from src.db.metadata import MetadataStore, DEFAULT_ACL_GROUPS, DEFAULT_PERSONAS


@pytest_asyncio.fixture
async def store():
    s = MetadataStore("sqlite+aiosqlite:///:memory:")
    await s.init()
    yield s


@pytest.mark.asyncio
async def test_seed_defaults_on_init(store):
    groups = await store.list_acl_groups(active_only=False)
    personas = await store.list_personas(active_only=False)
    assert len(groups) == len(DEFAULT_ACL_GROUPS)
    assert len(personas) == len(DEFAULT_PERSONAS)
    names = {g.name for g in groups}
    assert "finance" in names and "clinical" in names
    alice = await store.get_persona("alice")
    assert alice is not None
    assert "finance" in alice.groups
    assert "clinical" in alice.groups


@pytest.mark.asyncio
async def test_seed_defaults_idempotent(store):
    await store.seed_access_control_defaults()
    await store.seed_access_control_defaults()
    groups = await store.list_acl_groups(active_only=False)
    personas = await store.list_personas(active_only=False)
    assert len(groups) == len(DEFAULT_ACL_GROUPS)
    assert len(personas) == len(DEFAULT_PERSONAS)


@pytest.mark.asyncio
async def test_update_acl_group(store):
    await store.update_acl_group("finance", display_name="Finance Team", description="Budget folks")
    g = await store.get_acl_group("finance")
    assert g.display_name == "Finance Team"
    assert g.description == "Budget folks"


@pytest.mark.asyncio
async def test_deactivate_acl_group(store):
    await store.set_acl_group_active("finance", False)
    active = await store.list_acl_groups(active_only=True)
    assert "finance" not in {g.name for g in active}
    all_groups = await store.list_acl_groups(active_only=False)
    assert "finance" in {g.name for g in all_groups}


@pytest.mark.asyncio
async def test_persona_crud(store):
    p = await store.add_persona(
        name="rita",
        display_name="Rita (Clinical)",
        role="Nurse",
        groups=["clinical", "medical"],
    )
    assert p is not None
    assert p.name == "rita"
    await store.update_persona("rita", groups=["clinical", "medical", "hipaa"])
    p2 = await store.get_persona("rita")
    assert "hipaa" in p2.groups
    await store.delete_persona("rita")
    assert await store.get_persona("rita") is None


@pytest.mark.asyncio
async def test_resolve_play_user_groups_persona(store):
    groups = await store.resolve_play_user_groups("mike")
    assert "finance" in groups
    assert "executives" in groups


@pytest.mark.asyncio
async def test_resolve_play_user_groups_custom(store):
    groups = await store.resolve_play_user_groups("clinical, medical")
    assert groups == ["clinical", "medical"]


@pytest.mark.asyncio
async def test_resolve_play_user_groups_all(store):
    assert await store.resolve_play_user_groups("") == ["ALL"]
    assert await store.resolve_play_user_groups("ALL") == ["ALL"]


@pytest.mark.asyncio
async def test_discover_orphan_acl_groups(store):
    await store.add_document(
        doc_id="d1",
        filename="x.pdf",
        doc_type="pdf",
        acl_groups=["finance", "brand_new_group"],
        chunk_count=1,
        uploaded_by="admin",
    )
    orphans = await store.discover_orphan_acl_groups()
    assert orphans == ["brand_new_group"]


@pytest.mark.asyncio
async def test_uncovered_document_groups(store):
    # Remove clinical from all personas so it becomes uncovered
    for p in await store.list_personas(active_only=False):
        new_groups = [g for g in (p.groups or []) if g != "clinical"]
        await store.update_persona(p.name, groups=new_groups)
    await store.add_document(
        doc_id="d1",
        filename="c.pdf",
        doc_type="pdf",
        acl_groups=["clinical"],
        chunk_count=1,
        uploaded_by="admin",
    )
    uncovered = await store.uncovered_document_groups()
    assert "clinical" in uncovered
