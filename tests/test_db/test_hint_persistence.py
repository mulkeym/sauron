import pytest
from src.db.metadata import MetadataStore
from src.db.hint_store import SchemaHint


def _hint(**kw):
    base = dict(scope_type="category", scope_value="OPM", hint_type="value_glossary",
                target_column="locname", payload={"TU": "Tampa"}, provenance="curated",
                confidence=1.0, created_by="tester")
    base.update(kw)
    return SchemaHint(**base)


async def _store(tmp_path):
    s = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await s.init()
    return s


@pytest.mark.asyncio
async def test_save_and_load_round_trip(tmp_path):
    store = await _store(tmp_path)
    await store.save_hint(_hint())
    loaded = await store.load_all_hints()
    assert len(loaded) == 1
    h = loaded[0]
    assert h.scope_type == "category" and h.scope_value == "OPM"
    assert h.hint_type == "value_glossary" and h.target_column == "locname"
    assert h.payload == {"TU": "Tampa"}
    assert h.provenance == "curated" and h.id is not None


@pytest.mark.asyncio
async def test_list_hints_for_scope(tmp_path):
    store = await _store(tmp_path)
    await store.save_hint(_hint(scope_value="OPM"))
    await store.save_hint(_hint(scope_value="DoD", payload={"X": "Y"}))
    opm = await store.list_hints_for_scope("category", "OPM")
    assert len(opm) == 1 and opm[0].payload == {"TU": "Tampa"}


@pytest.mark.asyncio
async def test_delete_hint(tmp_path):
    store = await _store(tmp_path)
    await store.save_hint(_hint())
    h = (await store.load_all_hints())[0]
    await store.delete_hint(h.id)
    assert await store.load_all_hints() == []
