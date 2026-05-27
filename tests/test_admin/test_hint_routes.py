import pytest
from src.db.metadata import MetadataStore
from src.db.hint_store import HintStore
from src.admin import routes as admin_routes


async def _wire(tmp_path, monkeypatch):
    ms = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await ms.init()
    hs = HintStore()
    monkeypatch.setattr("src.api.routes_ingest.get_metadata_store", lambda: ms)
    monkeypatch.setattr("src.api.routes_ingest.get_hint_store", lambda: hs)
    return ms, hs


@pytest.mark.asyncio
async def test_create_and_list_hint(tmp_path, monkeypatch):
    ms, hs = await _wire(tmp_path, monkeypatch)
    hid = await admin_routes.create_hint_impl(
        scope_type="category", scope_value="OPM", hint_type="value_glossary",
        target_column="locname", payload={"TU": "Tampa"}, created_by="admin")
    assert hid is not None
    # registered live (no restart)
    assert hs.for_scope("category", "OPM")[0].payload == {"TU": "Tampa"}
    # persisted
    assert len(await ms.load_all_hints()) == 1


@pytest.mark.asyncio
async def test_bulk_import_glossary(tmp_path, monkeypatch):
    ms, hs = await _wire(tmp_path, monkeypatch)
    n = await admin_routes.bulk_import_hints_impl([
        {"scope_type": "category", "scope_value": "OPM", "hint_type": "value_glossary",
         "target_column": "locname", "payload": {"TU": "Tampa", "RUS": "Rest of U.S."}},
        {"scope_type": "category", "scope_value": "OPM", "hint_type": "table_note",
         "target_column": None, "payload": {"text": "OPM GS pay"}},
    ], created_by="admin")
    assert n == 2
    assert len(await ms.load_all_hints()) == 2
    assert len(hs.for_scope("category", "OPM")) == 2
