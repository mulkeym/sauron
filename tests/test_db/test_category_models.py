import pytest
import pytest_asyncio
from src.db.metadata import MetadataStore


@pytest_asyncio.fixture
async def store():
    s = MetadataStore("sqlite+aiosqlite:///:memory:")
    await s.init()
    yield s


@pytest.mark.asyncio
async def test_add_and_get_category(store):
    await store.add_category(name="finance_policies", description="Finance policy documents", acl_groups=["finance", "executives"], routing_keywords=["expense", "budget", "revenue"])
    cat = await store.get_category("finance_policies")
    assert cat is not None
    assert cat.name == "finance_policies"
    assert cat.acl_groups == ["finance", "executives"]


@pytest.mark.asyncio
async def test_list_categories(store):
    await store.add_category(name="finance", description="Finance", acl_groups=["finance"], routing_keywords=[])
    await store.add_category(name="it", description="IT", acl_groups=["it_support"], routing_keywords=[])
    cats = await store.list_categories()
    assert len(cats) == 2


@pytest.mark.asyncio
async def test_add_and_list_proposals(store):
    await store.add_proposal(proposed_name="legal_compliance", proposed_description="Legal docs", proposed_acl_groups=["legal"], proposed_keywords=["contract", "compliance"], proposed_by="system")
    proposals = await store.list_proposals(status="pending")
    assert len(proposals) == 1
    assert proposals[0].proposed_name == "legal_compliance"


@pytest.mark.asyncio
async def test_approve_proposal(store):
    await store.add_proposal(proposed_name="legal", proposed_description="Legal", proposed_acl_groups=["legal"], proposed_keywords=["legal"], proposed_by="system")
    proposals = await store.list_proposals(status="pending")
    proposal_id = proposals[0].id
    await store.approve_proposal(proposal_id, approved_by="admin")
    assert len(await store.list_proposals(status="pending")) == 0
    cat = await store.get_category("legal")
    assert cat is not None


@pytest.mark.asyncio
async def test_reject_proposal(store):
    await store.add_proposal(proposed_name="spam", proposed_description="Bad", proposed_acl_groups=[], proposed_keywords=[], proposed_by="system")
    proposals = await store.list_proposals(status="pending")
    await store.reject_proposal(proposals[0].id, rejected_by="admin")
    assert len(await store.list_proposals(status="pending")) == 0
    assert len(await store.list_proposals(status="rejected")) == 1
