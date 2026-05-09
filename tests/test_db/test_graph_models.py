import pytest
import pytest_asyncio
from src.db.metadata import MetadataStore


@pytest_asyncio.fixture
async def store():
    s = MetadataStore("sqlite+aiosqlite:///:memory:")
    await s.init()
    yield s


@pytest.mark.asyncio
async def test_add_and_search_entity(store):
    entity_id = await store.add_entity(name="Mike", entity_type="person", first_seen_doc_id="doc-1")
    assert entity_id is not None
    results = await store.search_entities("Mike")
    assert len(results) >= 1
    assert results[0].name == "Mike"


@pytest.mark.asyncio
async def test_add_entity_deduplicates(store):
    id1 = await store.add_entity(name="Policy 4.2", entity_type="policy", first_seen_doc_id="doc-1")
    id2 = await store.add_entity(name="Policy 4.2", entity_type="policy", first_seen_doc_id="doc-2")
    assert id1 == id2


@pytest.mark.asyncio
async def test_add_mention(store):
    entity_id = await store.add_entity(name="Mike", entity_type="person", first_seen_doc_id="doc-1")
    await store.add_mention(entity_id=entity_id, doc_id="doc-1", chunk_index=0, context_snippet="Mike asked about...")
    await store.add_mention(entity_id=entity_id, doc_id="doc-2", chunk_index=3, context_snippet="Mike reviewed...")
    details = await store.get_entity_details(entity_id)
    assert len(details["mentions"]) == 2


@pytest.mark.asyncio
async def test_add_relationship(store):
    e1 = await store.add_entity(name="Policy 4.2", entity_type="policy", first_seen_doc_id="doc-1")
    e2 = await store.add_entity(name="expense reporting", entity_type="project", first_seen_doc_id="doc-1")
    await store.add_relationship(source_entity_id=e1, target_entity_id=e2, relationship_type="governs", doc_id="doc-1", context_snippet="Policy 4.2 governs expense reporting")
    details = await store.get_entity_details(e1)
    assert len(details["relationships"]) == 1
    assert details["relationships"][0]["related_entity"] == "expense reporting"


@pytest.mark.asyncio
async def test_list_entities_by_type(store):
    await store.add_entity(name="Mike", entity_type="person", first_seen_doc_id="doc-1")
    await store.add_entity(name="Policy 4.2", entity_type="policy", first_seen_doc_id="doc-1")
    await store.add_entity(name="Sarah", entity_type="person", first_seen_doc_id="doc-2")
    people = await store.list_entities(entity_type="person")
    assert len(people) == 2


@pytest.mark.asyncio
async def test_search_entities_partial_match(store):
    await store.add_entity(name="Policy 4.2", entity_type="policy", first_seen_doc_id="doc-1")
    await store.add_entity(name="Policy 5.1", entity_type="policy", first_seen_doc_id="doc-1")
    results = await store.search_entities("policy")
    assert len(results) == 2


@pytest.mark.asyncio
async def test_delete_entities_for_doc(store):
    e1 = await store.add_entity(name="Mike", entity_type="person", first_seen_doc_id="doc-1")
    await store.add_mention(entity_id=e1, doc_id="doc-1", chunk_index=0, context_snippet="Mike")
    await store.delete_entities_for_doc("doc-1")
    details = await store.get_entity_details(e1)
    assert len(details["mentions"]) == 0
