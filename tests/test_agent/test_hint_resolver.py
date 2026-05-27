from types import SimpleNamespace
from src.db.hint_store import HintStore, SchemaHint
from src.db.schema_registry import TableSchema, ColumnSchema
from src.agent.strategies.hint_resolver import resolve_hints, ResolvedHints


def _schema():
    return TableSchema(database="spreadsheets", table="doc_x_all_gs",
                       columns=[ColumnSchema("locname", "VARCHAR"), ColumnSchema("grade", "DOUBLE")],
                       description="GS pay", acl_groups=["executives"])


def _doc(category="OPM", dataset_id=0):
    return SimpleNamespace(doc_id="x", category=category, dataset_id=dataset_id)


def _hint(**kw):
    base = dict(scope_type="category", scope_value="OPM", hint_type="value_glossary",
                target_column="locname", payload={"TU": "Tampa"}, provenance="curated", confidence=1.0)
    base.update(kw)
    return SchemaHint(**base)


def test_value_glossary_resolved_for_matching_column():
    store = HintStore(); store.register(_hint())
    r = resolve_hints(_schema(), _doc(), store)
    assert r.column_glossaries == {"locname": {"TU": "Tampa"}}
    assert r.column_notes == {} and r.table_notes == []


def test_hint_dropped_when_column_absent():
    store = HintStore(); store.register(_hint(target_column="nonexistent"))
    r = resolve_hints(_schema(), _doc(), store)
    assert r.column_glossaries == {}


def test_curated_overrides_auto_same_target():
    store = HintStore()
    store.register(_hint(provenance="auto", confidence=0.5, payload={"TU": "WRONG"}))
    store.register(_hint(provenance="curated", confidence=1.0, payload={"TU": "Tampa"}))
    r = resolve_hints(_schema(), _doc(), store)
    assert r.column_glossaries == {"locname": {"TU": "Tampa"}}


def test_category_and_dataset_scopes_merge():
    store = HintStore()
    store.register(_hint(scope_type="category", scope_value="OPM", target_column="locname"))
    store.register(_hint(scope_type="dataset", scope_value="7", hint_type="column_note",
                         target_column="grade", payload={"text": "pay grade level"}))
    r = resolve_hints(_schema(), _doc(category="OPM", dataset_id=7), store)
    assert r.column_glossaries == {"locname": {"TU": "Tampa"}}
    assert r.column_notes == {"grade": "pay grade level"}


def test_table_notes_collected():
    store = HintStore()
    store.register(_hint(hint_type="table_note", target_column=None, payload={"text": "OPM 2022 pay"}))
    r = resolve_hints(_schema(), _doc(), store)
    assert r.table_notes == ["OPM 2022 pay"]


def test_missing_doc_record_returns_empty():
    store = HintStore(); store.register(_hint())
    r = resolve_hints(_schema(), None, store)
    assert r.column_glossaries == {} and r.column_notes == {} and r.table_notes == []
