from src.db.hint_store import SchemaHint, HintStore


def _hint(scope_value="OPM", hint_type="value_glossary", target_column="locname",
          payload=None, provenance="curated", confidence=1.0):
    return SchemaHint(
        scope_type="category", scope_value=scope_value, hint_type=hint_type,
        target_column=target_column, payload=payload or {"TU": "Tampa"},
        provenance=provenance, confidence=confidence,
    )


def test_register_and_for_scope():
    store = HintStore()
    h = _hint()
    store.register(h)
    assert store.for_scope("category", "OPM") == [h]
    assert store.for_scope("category", "OTHER") == []
    assert store.for_scope("dataset", "OPM") == []


def test_for_scope_returns_all_matching():
    store = HintStore()
    a = _hint(target_column="locname")
    b = _hint(hint_type="table_note", target_column=None, payload={"text": "OPM pay data"})
    store.register(a)
    store.register(b)
    assert set(map(id, store.for_scope("category", "OPM"))) == {id(a), id(b)}


def test_clear():
    store = HintStore()
    store.register(_hint())
    store.clear()
    assert store.for_scope("category", "OPM") == []
