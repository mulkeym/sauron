import pytest
from src.db.metadata import MetadataStore
from src.db.hint_store import HintStore, SchemaHint
from src.ingestion.tabular_ingest import populate_hint_store


@pytest.mark.asyncio
async def test_populate_hint_store_loads_persisted(tmp_path):
    ms = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await ms.init()
    await ms.save_hint(SchemaHint(
        scope_type="category", scope_value="OPM", hint_type="value_glossary",
        target_column="locname", payload={"TU": "Tampa"},
    ))
    store = HintStore()
    n = await populate_hint_store(ms, store)
    assert n == 1
    assert store.for_scope("category", "OPM")[0].payload == {"TU": "Tampa"}
