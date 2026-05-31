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


from types import SimpleNamespace
from src.db.hint_store import SchemaHint
from src.admin.routes import _build_hints_view


def _hints():
    return [
        SchemaHint(scope_type="category", scope_value="payroll_compensation",
                   hint_type="value_glossary", target_column="locname",
                   payload={"MFL": "Miami", "AK": "State of Alaska"}),
        SchemaHint(scope_type="category", scope_value="payroll_compensation",
                   hint_type="column_note", target_column="locname",
                   payload={"text": "OPM locality code"}),
        SchemaHint(scope_type="dataset", scope_value="2",
                   hint_type="table_note", target_column=None,
                   payload={"text": "Military pay"}),
    ]


def test_build_hints_view_groups_and_labels():
    datasets = [SimpleNamespace(id=2, name="Military Pay")]
    groups = _build_hints_view(_hints(), datasets)
    labels = [g["scope_label"] for g in groups]
    assert labels == ["category = payroll_compensation", "dataset = Military Pay (id 2)"]


def test_build_hints_view_glossary_entries_sorted_and_counted():
    groups = _build_hints_view(_hints(), [SimpleNamespace(id=2, name="Military Pay")])
    cat = next(g for g in groups if g["scope_label"].startswith("category"))
    types = [h["hint_type"] for h in cat["hints"]]
    assert types == ["column_note", "value_glossary"]
    gloss = next(h for h in cat["hints"] if h["hint_type"] == "value_glossary")
    assert gloss["count"] == 2
    assert gloss["entries"] == [{"code": "AK", "meaning": "State of Alaska"},
                                {"code": "MFL", "meaning": "Miami"}]
    note = next(h for h in cat["hints"] if h["hint_type"] == "column_note")
    assert note["text"] == "OPM locality code"


def test_build_hints_view_unknown_dataset_falls_back_to_id():
    h = [SchemaHint(scope_type="dataset", scope_value="9", hint_type="table_note",
                    target_column=None, payload={"text": "x"})]
    groups = _build_hints_view(h, [])
    assert groups[0]["scope_label"] == "dataset = 9"


def test_hints_page_renders():
    from unittest.mock import patch, AsyncMock
    from fastapi.testclient import TestClient
    from src.main import create_app
    client = TestClient(create_app())
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        store.load_all_hints.return_value = [
            SchemaHint(scope_type="category", scope_value="payroll_compensation",
                       hint_type="value_glossary", target_column="locname",
                       payload={"AK": "State of Alaska"}),
        ]
        store.list_datasets.return_value = []
        mock_get.return_value = store
        resp = client.get("/admin/hints")
    assert resp.status_code == 200
    assert "payroll_compensation" in resp.text
    assert "State of Alaska" in resp.text
