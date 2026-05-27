# Extensible Table Hints Implementation Plan (Phase 1: curated foundation)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let curated domain knowledge ("table hints" — e.g. a locality code→name glossary) be attached to a source/collection and injected into the text-to-SQL schema prompt, so the model filters on the right codes.

**Architecture:** A `SchemaHint` record (persisted in metadata.db, scoped to a `category` or `dataset`) is loaded at startup into an in-memory `HintStore` (parallels `SchemaRegistry`). At query time the structured-retrieval path resolves the hints applicable to each in-scope table (via its owning `DocumentRecord`'s category/dataset), and a pure injector annotates the schema prompt. Fully additive + fail-open: with no hints, the prompt is byte-identical to today.

**Tech Stack:** Python 3, pytest, SQLAlchemy (async), pydantic/dataclasses, DuckDB. No new dependencies.

Spec: `docs/superpowers/specs/2026-05-27-extensible-table-hints-design.md`.

---

## Task 1: `SchemaHint` dataclass + `HintStore`

**Files:**
- Create: `src/db/hint_store.py`
- Test: `tests/test_db/test_hint_store.py`

`SchemaHint` is the in-memory/dataclass form (mirrors `TableSchema` in `src/db/schema_registry.py`). `HintStore` mirrors `SchemaRegistry`: in-memory, indexed by `(scope_type, scope_value)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_db/test_hint_store.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_db/test_hint_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.db.hint_store'`.

- [ ] **Step 3: Implement `hint_store.py`**

Create `src/db/hint_store.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SchemaHint:
    """One piece of curated (or, later, auto/learned) domain knowledge attached to
    a source/collection, injected into the text-to-SQL schema prompt.

    scope_type/scope_value bind the hint to a ``category`` (name) or ``dataset``
    (id as string). ``target_column`` names the column the hint applies to
    (None for ``table_note``); matched by name across every table in scope.
    ``hint_type`` is one of ``value_glossary`` | ``column_note`` | ``table_note``.
    For value_glossary, ``payload`` is ``{code: meaning}``; for notes,
    ``{"text": ...}``. ``provenance`` is ``curated`` | ``auto`` | ``learned``."""
    scope_type: str
    scope_value: str
    hint_type: str
    target_column: str | None
    payload: dict
    provenance: str = "curated"
    confidence: float = 1.0
    id: int | None = None
    created_at: datetime | None = None
    created_by: str = ""


class HintStore:
    """In-memory store of SchemaHints, indexed by (scope_type, scope_value).
    Parallels SchemaRegistry; loaded at startup from the metadata store."""

    def __init__(self):
        self._by_scope: dict[tuple[str, str], list[SchemaHint]] = {}

    def register(self, hint: SchemaHint) -> None:
        self._by_scope.setdefault((hint.scope_type, hint.scope_value), []).append(hint)

    def for_scope(self, scope_type: str, scope_value: str) -> list[SchemaHint]:
        return list(self._by_scope.get((scope_type, scope_value), []))

    def clear(self) -> None:
        self._by_scope.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_db/test_hint_store.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/db/hint_store.py tests/test_db/test_hint_store.py
git commit -m "feat: SchemaHint dataclass + in-memory HintStore"
```

End every commit in this plan with:
```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 2: ORM model + `MetadataStore` persistence

**Files:**
- Modify: `src/db/models.py` (add `SchemaHintRecord` after `RegisteredSchema`, ~line 237)
- Modify: `src/db/metadata.py` (add CRUD methods after `delete_schema`, ~line 575)
- Test: `tests/test_db/test_hint_persistence.py`

`models.py` already imports `Mapped, mapped_column, String, JSON, DateTime, Float, Integer, UniqueConstraint, datetime, timezone` for the other models (`RegisteredSchema` uses `String/JSON/DateTime`; confirm `Float` is imported — if not, add it to the existing `from sqlalchemy import ...` line). `create_all` in `MetadataStore.init` creates new tables automatically, so no manual migration entry is needed for a brand-new table.

- [ ] **Step 1: Write the failing test**

Create `tests/test_db/test_hint_persistence.py`:

```python
import pytest
from src.db.metadata import MetadataStore
from src.db.hint_store import SchemaHint


@pytest.fixture
async def store(tmp_path):
    s = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await s.init()
    return s


def _hint(**kw):
    base = dict(scope_type="category", scope_value="OPM", hint_type="value_glossary",
                target_column="locname", payload={"TU": "Tampa"}, provenance="curated",
                confidence=1.0, created_by="tester")
    base.update(kw)
    return SchemaHint(**base)


@pytest.mark.asyncio
async def test_save_and_load_round_trip(store):
    await store.save_hint(_hint())
    loaded = await store.load_all_hints()
    assert len(loaded) == 1
    h = loaded[0]
    assert h.scope_type == "category" and h.scope_value == "OPM"
    assert h.hint_type == "value_glossary" and h.target_column == "locname"
    assert h.payload == {"TU": "Tampa"}
    assert h.provenance == "curated" and h.id is not None


@pytest.mark.asyncio
async def test_list_hints_for_scope(store):
    await store.save_hint(_hint(scope_value="OPM"))
    await store.save_hint(_hint(scope_value="DoD", payload={"X": "Y"}))
    opm = await store.list_hints_for_scope("category", "OPM")
    assert len(opm) == 1 and opm[0].payload == {"TU": "Tampa"}


@pytest.mark.asyncio
async def test_delete_hint(store):
    await store.save_hint(_hint())
    h = (await store.load_all_hints())[0]
    await store.delete_hint(h.id)
    assert await store.load_all_hints() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_db/test_hint_persistence.py -v`
Expected: FAIL — `AttributeError: 'MetadataStore' object has no attribute 'save_hint'`.

- [ ] **Step 3a: Add the ORM model**

In `src/db/models.py`, immediately AFTER the `RegisteredSchema` class (after its `__table_args__` line, ~237), add:

```python
class SchemaHintRecord(Base):
    __tablename__ = "schema_hints"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    scope_type: Mapped[str] = mapped_column(String, nullable=False)   # "category" | "dataset"
    scope_value: Mapped[str] = mapped_column(String, nullable=False)
    hint_type: Mapped[str] = mapped_column(String, nullable=False)    # value_glossary | column_note | table_note
    target_column: Mapped[str] = mapped_column(String, default="")    # "" == applies to table (table_note)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    provenance: Mapped[str] = mapped_column(String, default="curated")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
```

If `Float` is not already in the `from sqlalchemy import ...` line at the top of `models.py`, add it. (`String, JSON, DateTime, Integer` are already imported for the existing models.)

- [ ] **Step 3b: Add the import + CRUD methods**

In `src/db/metadata.py`, add `SchemaHintRecord` to the model import line (`from src.db.models import Base, DocumentRecord, ... RegisteredSchema`) → append `, SchemaHintRecord`. Then add the import for the dataclass at the existing schema import: change `from src.db.schema_registry import TableSchema, ColumnSchema` to also import nothing new (SchemaHint comes from hint_store). Add at the top with the other imports: `from src.db.hint_store import SchemaHint`.

Add these methods immediately AFTER `delete_schema` (~line 575):

```python
    def _hint_from_record(self, r) -> SchemaHint:
        return SchemaHint(
            id=r.id, scope_type=r.scope_type, scope_value=r.scope_value,
            hint_type=r.hint_type, target_column=(r.target_column or None),
            payload=r.payload or {}, provenance=r.provenance,
            confidence=r.confidence, created_by=r.created_by, created_at=r.created_at,
        )

    async def save_hint(self, hint: SchemaHint) -> int:
        """Persist a SchemaHint; returns its row id."""
        async with self.session_factory() as session:
            rec = SchemaHintRecord(
                scope_type=hint.scope_type, scope_value=hint.scope_value,
                hint_type=hint.hint_type, target_column=hint.target_column or "",
                payload=hint.payload or {}, provenance=hint.provenance,
                confidence=hint.confidence, created_by=hint.created_by,
            )
            session.add(rec)
            await session.commit()
            await session.refresh(rec)
            return rec.id

    async def load_all_hints(self) -> list[SchemaHint]:
        async with self.session_factory() as session:
            result = await session.execute(select(SchemaHintRecord))
            return [self._hint_from_record(r) for r in result.scalars().all()]

    async def list_hints_for_scope(self, scope_type: str, scope_value: str) -> list[SchemaHint]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(SchemaHintRecord).where(
                    SchemaHintRecord.scope_type == scope_type,
                    SchemaHintRecord.scope_value == scope_value,
                )
            )
            return [self._hint_from_record(r) for r in result.scalars().all()]

    async def delete_hint(self, hint_id: int) -> None:
        async with self.session_factory() as session:
            await session.execute(delete(SchemaHintRecord).where(SchemaHintRecord.id == hint_id))
            await session.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_db/test_hint_persistence.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/db/models.py src/db/metadata.py tests/test_db/test_hint_persistence.py
git commit -m "feat: persist SchemaHints (schema_hints table + MetadataStore CRUD)"
```

---

## Task 3: `populate_hint_store` + `get_hint_store` + startup wiring

**Files:**
- Modify: `src/ingestion/tabular_ingest.py` (add `populate_hint_store` after `populate_schema_registry`, ~line 185)
- Modify: `src/api/routes_ingest.py` (add `get_hint_store` after `get_schema_registry`, ~line 36)
- Modify: `src/main.py` (lifespan, after the schema-registry load, ~line 60)
- Test: `tests/test_ingestion/test_populate_hint_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion/test_populate_hint_store.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_populate_hint_store.py -v`
Expected: FAIL — `ImportError: cannot import name 'populate_hint_store'`.

- [ ] **Step 3a: Add `populate_hint_store`**

In `src/ingestion/tabular_ingest.py`, immediately AFTER the `populate_schema_registry` function (~line 185), add:

```python
async def populate_hint_store(metadata_store, hint_store) -> int:
    """Load all persisted SchemaHints into the in-memory HintStore. Returns count."""
    hints = await metadata_store.load_all_hints()
    for h in hints:
        hint_store.register(h)
    return len(hints)
```

- [ ] **Step 3b: Add `get_hint_store` singleton**

In `src/api/routes_ingest.py`, after `get_schema_registry` (~line 36), add (and add `from src.db.hint_store import HintStore` near the existing `from src.db.schema_registry import SchemaRegistry`):

```python
_hint_store = None

def get_hint_store():
    global _hint_store
    if _hint_store is None:
        _hint_store = HintStore()
    return _hint_store
```

- [ ] **Step 3c: Wire startup load**

In `src/main.py` lifespan, immediately AFTER the schema-registry load block (after line ~60, before `yield`), add:

```python
    # Load persisted table hints into the in-memory hint store
    try:
        from src.api.routes_ingest import get_hint_store
        from src.ingestion.tabular_ingest import populate_hint_store
        hn = await populate_hint_store(store, get_hint_store())
        logging.getLogger(__name__).info(f"Loaded {hn} table hint(s) into the hint store")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Hint store load deferred: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion/test_populate_hint_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_ingest.py src/api/routes_ingest.py src/main.py tests/test_ingestion/test_populate_hint_store.py
git commit -m "feat: populate_hint_store + get_hint_store singleton + startup load"
```

---

## Task 4: Pure resolver `resolve_hints`

**Files:**
- Create: `src/agent/strategies/hint_resolver.py`
- Test: `tests/test_agent/test_hint_resolver.py`

`resolve_hints(table_schema, doc_record, hint_store) -> ResolvedHints` is pure (caller supplies the doc record). It gathers hints from the doc's `category` and `dataset_id` scopes, dedups per `(hint_type, target_column)` with precedence `curated > learned > auto` (then higher confidence, then most recent), drops value/column hints whose `target_column` is not a column of `table_schema`, and returns a compact structure.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent/test_hint_resolver.py`:

```python
from datetime import datetime, timezone
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_hint_resolver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agent.strategies.hint_resolver'`.

- [ ] **Step 3: Implement the resolver**

Create `src/agent/strategies/hint_resolver.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field

_PROV_RANK = {"curated": 3, "learned": 2, "auto": 1}


@dataclass
class ResolvedHints:
    column_glossaries: dict[str, dict[str, str]] = field(default_factory=dict)  # col -> {code: meaning}
    column_notes: dict[str, str] = field(default_factory=dict)                  # col -> note
    table_notes: list[str] = field(default_factory=list)


def _better(a, b) -> bool:
    """True if hint ``a`` outranks hint ``b`` (provenance, then confidence, then recency)."""
    ra, rb = _PROV_RANK.get(a.provenance, 0), _PROV_RANK.get(b.provenance, 0)
    if ra != rb:
        return ra > rb
    if (a.confidence or 0) != (b.confidence or 0):
        return (a.confidence or 0) > (b.confidence or 0)
    return (a.created_at or 0) and (b.created_at or 0) and a.created_at >= b.created_at


def resolve_hints(table_schema, doc_record, hint_store) -> ResolvedHints:
    """Hints applicable to ``table_schema`` given its owning ``doc_record``. Pure;
    fail-safe (missing doc / malformed payloads are skipped, never raised)."""
    out = ResolvedHints()
    if doc_record is None:
        return out
    col_names = {c.name for c in table_schema.columns}

    scopes = [("category", getattr(doc_record, "category", "") or ""),
              ("dataset", str(getattr(doc_record, "dataset_id", "") or ""))]
    hints = []
    for st, sv in scopes:
        if sv:
            hints.extend(hint_store.for_scope(st, sv))

    # Dedup non-table hints per (hint_type, target_column), keeping the best.
    best: dict[tuple, object] = {}
    for h in hints:
        try:
            if h.hint_type == "table_note":
                text = (h.payload or {}).get("text", "")
                if text:
                    out.table_notes.append(text)
                continue
            if h.target_column not in col_names:
                continue
            key = (h.hint_type, h.target_column)
            if key not in best or _better(h, best[key]):
                best[key] = h
        except Exception:
            continue

    for (hint_type, col), h in best.items():
        if hint_type == "value_glossary" and isinstance(h.payload, dict):
            out.column_glossaries[col] = {str(k): str(v) for k, v in h.payload.items()}
        elif hint_type == "column_note":
            text = (h.payload or {}).get("text", "")
            if text:
                out.column_notes[col] = text
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent/test_hint_resolver.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/hint_resolver.py tests/test_agent/test_hint_resolver.py
git commit -m "feat: pure resolve_hints — applicable hints for a table via its document scope"
```

---

## Task 5: Prompt injection in `schema_prompt_with_values`

**Files:**
- Modify: `src/ingestion/tabular_store.py` (`schema_prompt_with_values`, ~line 216)
- Test: `tests/test_ingestion/test_tabular_store.py` (append)

Add an optional `hints: dict[str, ResolvedHints] | None` keyed by table name. When a table has resolved hints: annotate the `values:` line with glossary meanings, append column notes to descriptions, and add a `Notes:` line for table notes. `hints=None` ⇒ byte-identical to today.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ingestion/test_tabular_store.py`:

```python
from src.agent.strategies.hint_resolver import ResolvedHints


def test_schema_prompt_hints_none_is_unchanged():
    con, table = _con_with_pay()
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    schema = schema_from_sheet("doc1", "Pay", cls, _pay_grid(), acl_groups=["ALL"])
    assert schema_prompt_with_values([schema], con, hints=None) == schema_prompt_with_values([schema], con)


def test_schema_prompt_annotates_glossary_values():
    con, table = _con_with_pay()
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    schema = schema_from_sheet("doc1", "Pay", cls, _pay_grid(), acl_groups=["ALL"])
    hints = {schema.table: ResolvedHints(column_glossaries={"grade": {"GS-12": "Senior"}})}
    prompt = schema_prompt_with_values([schema], con, hints=hints)
    assert "GS-12 (Senior)" in prompt


def test_schema_prompt_adds_column_and_table_notes():
    con, table = _con_with_pay()
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    schema = schema_from_sheet("doc1", "Pay", cls, _pay_grid(), acl_groups=["ALL"])
    hints = {schema.table: ResolvedHints(
        column_notes={"grade": "GS pay grade"}, table_notes=["OPM 2022 GS pay"])}
    prompt = schema_prompt_with_values([schema], con, hints=hints)
    assert "GS pay grade" in prompt
    assert "Notes: OPM 2022 GS pay" in prompt
```

(`_con_with_pay`, `_pay_grid`, `schema_from_sheet`, `SheetClassification` are already imported/defined in this test file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_tabular_store.py -k "schema_prompt_hints or schema_prompt_annotates or schema_prompt_adds" -v`
Expected: FAIL — `schema_prompt_with_values() got an unexpected keyword argument 'hints'`.

- [ ] **Step 3: Implement injection**

In `src/ingestion/tabular_store.py`, REPLACE the entire `schema_prompt_with_values` function (currently ~lines 216-234) with:

```python
def schema_prompt_with_values(schemas, con, max_distinct: int = 100, hints=None) -> str:
    """Render registered schemas for the text-to-SQL prompt, appending the
    distinct values of low-cardinality VARCHAR columns so the LLM filters on
    real category codes (the gap that made 'Rest of U.S.' match 0 rows).

    ``hints`` (optional) maps table name -> ResolvedHints; when present its
    glossaries annotate the values list (``CODE (meaning)``), column notes append
    to the column description, and table notes render as a ``Notes:`` line. With
    ``hints=None`` (or no entry for a table) output is byte-identical to before.
    """
    parts = []
    for s in schemas:
        th = (hints or {}).get(s.table)
        glossaries = th.column_glossaries if th else {}
        col_notes = th.column_notes if th else {}
        col_lines = []
        for c in s.columns:
            desc = c.description
            if c.name in col_notes:
                desc = f"{desc} — {col_notes[c.name]}" if desc else col_notes[c.name]
            line = f"  - {c.name} ({c.dtype}): {desc}"
            if c.dtype == "VARCHAR":
                vals = distinct_values(con, s.table, c.name, max_distinct)
                if vals:
                    gloss = glossaries.get(c.name, {})
                    rendered = [f"{v} ({gloss[str(v)]})" if str(v) in gloss else str(v) for v in vals]
                    line += " | values: " + ", ".join(rendered)
            col_lines.append(line)
        header = f"Table: {s.table}\nDescription: {s.description}"
        if th and th.table_notes:
            header += "\nNotes: " + "; ".join(th.table_notes)
        parts.append(header + "\nColumns:\n" + "\n".join(col_lines))
    return "\n\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ingestion/test_tabular_store.py -v`
Expected: PASS — the 3 new tests plus all pre-existing tabular_store tests (the `hints=None` test guards the unchanged path).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_store.py tests/test_ingestion/test_tabular_store.py
git commit -m "feat: inject resolved table hints into the text-to-SQL schema prompt"
```

---

## Task 6: Resolve + thread hints through the structured path

**Files:**
- Modify: `src/agent/strategies/structured.py` (add `resolve_hints_for_schemas`; add `hints` param to `run_structured_lookup` + `structured_sql_rows`)
- Modify: `src/agent/strategies/analytical.py` (`retrieve_analytical` resolves + passes hints)
- Test: `tests/test_agent/test_structured_hints_integration.py`

`run_structured_lookup` (sync) calls `schema_prompt_with_values(schemas, con)` at line ~116; `structured_sql_rows` (sync) at line ~143. Both gain an optional `hints` param forwarded to `schema_prompt_with_values`. A new async helper resolves the per-table hint map (async because it lists documents) so the sync core stays pure.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent/test_structured_hints_integration.py`:

```python
import pytest
from types import SimpleNamespace
from src.db.hint_store import HintStore, SchemaHint
from src.db.schema_registry import TableSchema, ColumnSchema
from src.agent.strategies.structured import resolve_hints_for_schemas


class _MS:
    def __init__(self, docs):
        self._docs = docs
    async def list_documents(self, user_groups=None):
        return self._docs


def _schema(table):
    return TableSchema(database="spreadsheets", table=table,
                       columns=[ColumnSchema("locname", "VARCHAR")],
                       description="d", acl_groups=["executives"])


@pytest.mark.asyncio
async def test_resolve_hints_for_schemas_maps_table_to_doc_scope():
    from src.ingestion.tabular_store import duckdb_table_name
    doc = SimpleNamespace(doc_id="abc", category="OPM", dataset_id=0)
    table = duckdb_table_name("abc", "all_gs")
    store = HintStore()
    store.register(SchemaHint(scope_type="category", scope_value="OPM",
                              hint_type="value_glossary", target_column="locname",
                              payload={"TU": "Tampa"}))
    out = await resolve_hints_for_schemas([_schema(table)], HintStoreAndMs := store, _MS([doc]))
    assert out[table].column_glossaries == {"locname": {"TU": "Tampa"}}


@pytest.mark.asyncio
async def test_resolve_hints_for_schemas_empty_when_no_owning_doc():
    store = HintStore()
    out = await resolve_hints_for_schemas([_schema("doc_ghost_all_gs")], store, _MS([]))
    assert out == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_structured_hints_integration.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_hints_for_schemas'`.

- [ ] **Step 3a: Add `resolve_hints_for_schemas` + thread `hints`**

In `src/agent/strategies/structured.py`:

Add near the top imports:
```python
from src.agent.strategies.hint_resolver import resolve_hints
```

Add this async helper (e.g. after the `tables_relevant_to` function, before `retrieve_structured`):

```python
async def resolve_hints_for_schemas(schemas, hint_store, metadata_store) -> dict:
    """Map each schema's table name -> ResolvedHints, by finding its owning
    document (table name prefix == duckdb_table_name(doc_id, "")) and resolving
    that document's category/dataset-scoped hints. Fail-open: returns {} on any
    error; tables with no owning doc or no hints are omitted."""
    from src.ingestion.tabular_store import duckdb_table_name
    try:
        docs = await metadata_store.list_documents()
    except Exception:
        return {}
    out = {}
    for s in schemas:
        owner = next((d for d in docs if s.table.startswith(duckdb_table_name(d.doc_id, ""))), None)
        if owner is None:
            continue
        rh = resolve_hints(s, owner, hint_store)
        if rh.column_glossaries or rh.column_notes or rh.table_notes:
            out[s.table] = rh
    return out
```

Change `run_structured_lookup` signature (line ~107) to accept `hints=None` and forward it:
```python
def run_structured_lookup(question: str, schemas, query_type: str,
                          gate: list | None = None, generate_fn=None, hints=None) -> StructuredLookupTrace:
```
and at its `schema_prompt_with_values(...)` call (line ~116) add `, hints=hints`:
```python
        trace.sql = generate_sql(schema_prompt_with_values(schemas, con, hints=hints), question,
                                 generate_fn=generate_fn)
```

Change `structured_sql_rows` signature (line ~132) to accept `hints=None` and forward it at its `schema_prompt_with_values(...)` call (line ~143) → add `, hints=hints`.

- [ ] **Step 3b: Resolve + pass hints in `retrieve_analytical`**

In `src/agent/strategies/analytical.py`, in `retrieve_analytical`, REPLACE the line:
```python
    trace = await asyncio.to_thread(run_structured_lookup, question, schemas, "analytical")
```
with:
```python
    from src.agent.strategies.structured import resolve_hints_for_schemas
    from src.api.routes_ingest import get_hint_store, get_metadata_store
    try:
        hints = await resolve_hints_for_schemas(schemas, get_hint_store(), get_metadata_store())
    except Exception:
        hints = None
    trace = await asyncio.to_thread(run_structured_lookup, question, schemas, "analytical", None, None, hints)
```

(`run_structured_lookup(question, schemas, query_type, gate, generate_fn, hints)` — positional `None, None, hints` fills `gate` and `generate_fn` with their defaults.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent/test_structured_hints_integration.py tests/test_agent/test_synthesizer.py -v`
Expected: the 2 new tests PASS; synthesizer tests unaffected.

Also run the structured strategy's own suite to confirm no regression:
Run: `python -m pytest tests/test_agent/ -k "structured or analytical" -q`
Expected: no NEW failures vs. master (pre-existing test_graph/lookup/sweep failures are unrelated — see roadmap memory).

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py src/agent/strategies/analytical.py tests/test_agent/test_structured_hints_integration.py
git commit -m "feat: resolve + inject table hints in the analytical structured-SQL path"
```

---

## Task 7: Admin authoring API + bulk import

**Files:**
- Modify: `src/admin/routes.py` (add hint CRUD + bulk-import endpoints; follow the existing category-management route style)
- Test: `tests/test_admin/test_hint_routes.py`

Expose minimal endpoints so an operator can create/list/delete a hint and bulk-import a glossary. These persist via `MetadataStore` AND register into the live `HintStore` so no restart is needed. Follow the auth/dependency pattern of the existing admin category routes in this file.

- [ ] **Step 1: Inspect the existing admin route pattern**

Run: `grep -n "add_category\|@router.post\|@router.get\|Depends\|require_admin\|async def" src/admin/routes.py | head -40`
Note the router object name, the admin-auth dependency, and how an existing POST route reads its body + calls the metadata store. Mirror that exact style (router name, auth dependency) in Step 3. (Do not invent a new auth scheme.)

- [ ] **Step 2: Write the failing test**

Create `tests/test_admin/test_hint_routes.py`:

```python
import pytest
from src.db.metadata import MetadataStore
from src.db.hint_store import HintStore
from src.admin import routes as admin_routes


@pytest.fixture
async def wired(tmp_path, monkeypatch):
    ms = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await ms.init()
    hs = HintStore()
    monkeypatch.setattr("src.api.routes_ingest.get_metadata_store", lambda: ms)
    monkeypatch.setattr("src.api.routes_ingest.get_hint_store", lambda: hs)
    return ms, hs


@pytest.mark.asyncio
async def test_create_and_list_hint(wired):
    ms, hs = wired
    hid = await admin_routes.create_hint_impl(
        scope_type="category", scope_value="OPM", hint_type="value_glossary",
        target_column="locname", payload={"TU": "Tampa"}, created_by="admin")
    assert hid is not None
    # registered live (no restart)
    assert hs.for_scope("category", "OPM")[0].payload == {"TU": "Tampa"}
    # persisted
    assert len(await ms.load_all_hints()) == 1


@pytest.mark.asyncio
async def test_bulk_import_glossary(wired):
    ms, hs = wired
    n = await admin_routes.bulk_import_hints_impl([
        {"scope_type": "category", "scope_value": "OPM", "hint_type": "value_glossary",
         "target_column": "locname", "payload": {"TU": "Tampa", "RUS": "Rest of U.S."}},
        {"scope_type": "category", "scope_value": "OPM", "hint_type": "table_note",
         "target_column": None, "payload": {"text": "OPM GS pay"}},
    ], created_by="admin")
    assert n == 2
    assert len(await ms.load_all_hints()) == 2
    assert len(hs.for_scope("category", "OPM")) == 2
```

- [ ] **Step 3: Implement the impl functions + thin routes**

In `src/admin/routes.py`, add two module-level async impl functions (testable without HTTP) and thin route wrappers that call them. Place the impls near the category-management code; place the routes beside the other admin routes using the SAME router object and admin-auth dependency you identified in Step 1.

```python
async def create_hint_impl(scope_type, scope_value, hint_type, target_column,
                           payload, provenance="curated", confidence=1.0, created_by=""):
    """Persist one SchemaHint and register it live. Returns its id."""
    from src.api.routes_ingest import get_metadata_store, get_hint_store
    from src.db.hint_store import SchemaHint
    ms = get_metadata_store()
    hint = SchemaHint(scope_type=scope_type, scope_value=scope_value, hint_type=hint_type,
                      target_column=target_column, payload=payload or {},
                      provenance=provenance, confidence=confidence, created_by=created_by)
    hint.id = await ms.save_hint(hint)
    get_hint_store().register(hint)
    return hint.id


async def bulk_import_hints_impl(items, created_by=""):
    """Persist + register a list of hint dicts (see create_hint_impl args).
    Returns the count imported. Skips malformed entries."""
    n = 0
    for it in items:
        try:
            await create_hint_impl(
                scope_type=it["scope_type"], scope_value=it["scope_value"],
                hint_type=it["hint_type"], target_column=it.get("target_column"),
                payload=it.get("payload") or {}, provenance=it.get("provenance", "curated"),
                confidence=it.get("confidence", 1.0), created_by=created_by)
            n += 1
        except Exception:
            continue
    return n


async def delete_hint_impl(hint_id):
    """Delete a hint by id and rebuild the live HintStore from persistence."""
    from src.api.routes_ingest import get_metadata_store, get_hint_store
    from src.ingestion.tabular_ingest import populate_hint_store
    await get_metadata_store().delete_hint(int(hint_id))
    store = get_hint_store()
    store.clear()
    await populate_hint_store(get_metadata_store(), store)
```

Then add thin route wrappers using the router + admin-auth dependency from Step 1, e.g.:

```python
@router.post("/hints")
async def create_hint_route(body: dict, _admin=Depends(<the admin dependency from Step 1>)):
    hid = await create_hint_impl(
        scope_type=body["scope_type"], scope_value=body["scope_value"],
        hint_type=body["hint_type"], target_column=body.get("target_column"),
        payload=body.get("payload") or {}, provenance=body.get("provenance", "curated"),
        confidence=body.get("confidence", 1.0), created_by=body.get("created_by", "admin"))
    return {"id": hid}


@router.post("/hints/bulk")
async def bulk_import_hints_route(body: dict, _admin=Depends(<the admin dependency from Step 1>)):
    return {"imported": await bulk_import_hints_impl(body.get("hints", []),
                                                     created_by=body.get("created_by", "admin"))}


@router.get("/hints")
async def list_hints_route(_admin=Depends(<the admin dependency from Step 1>)):
    from src.api.routes_ingest import get_metadata_store
    hints = await get_metadata_store().load_all_hints()
    return {"hints": [vars(h) | {"created_at": str(h.created_at)} for h in hints]}


@router.delete("/hints/{hint_id}")
async def delete_hint_route(hint_id: int, _admin=Depends(<the admin dependency from Step 1>)):
    await delete_hint_impl(hint_id)
    return {"ok": True}
```

Replace `<the admin dependency from Step 1>` with the actual dependency callable used by neighboring admin routes (e.g. `require_admin`). Match the actual `router` name in the file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_admin/test_hint_routes.py -v`
Expected: PASS (2 tests).

Confirm the module still imports (routes wired without syntax error):
Run: `python -c "import src.admin.routes"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add src/admin/routes.py tests/test_admin/test_hint_routes.py
git commit -m "feat: admin endpoints to create/list/delete + bulk-import table hints (live + persisted)"
```

---

## Task 8: Regression + end-to-end verification + memory

**Files:** none (verification); then memory update

- [ ] **Step 1: Run the affected suites**

Run: `python -m pytest tests/test_db/ tests/test_ingestion/test_tabular_store.py tests/test_ingestion/test_populate_hint_store.py tests/test_agent/test_hint_resolver.py tests/test_agent/test_structured_hints_integration.py tests/test_admin/test_hint_routes.py tests/test_agent/test_synthesizer.py -q`
Expected: all new tests pass; no new failures. (`tests/test_ingestion` and `tests/test_agent` carry known pre-existing/environmental failures — numpy/sklearn ABI, OpenAI 401, test_graph/lookup/sweep harness issues — documented in the roadmap memory; confirm the failing set is unchanged.)

- [ ] **Step 2: Rebuild + redeploy, then end-to-end smoke**

```bash
docker compose up -d --build api
```

Then, inside the container, curate the OPM locality glossary scoped to the GS docs' category and confirm the prompt + SQL improve. Replace `<CATEGORY>` with the real category of the GS docs (find it: `docker compose exec -T api python -c "import asyncio;from src.api.routes_ingest import get_metadata_store as g;print(asyncio.run(g().list_documents.__self__.list_documents()))"` — or inspect a GS DocumentRecord's `.category`). Then:

```bash
docker compose exec -T api python -c "
import asyncio
from src.admin.routes import bulk_import_hints_impl
asyncio.run(bulk_import_hints_impl([
  {'scope_type':'category','scope_value':'<CATEGORY>','hint_type':'value_glossary',
   'target_column':'locname','payload':{'TU':'Tampa, FL','RUS':'Rest of U.S.'}}
], created_by='smoke'))
print('imported')
"
```

Restart is NOT required (impl registers live), but the in-container check below uses a fresh process, so re-populate the store in that process. Confirm the glossary now annotates the prompt and the model targets the right code:

```bash
docker compose exec -T api python -c "
import asyncio
from src.api.routes_ingest import get_metadata_store, get_hint_store
from src.ingestion.tabular_ingest import populate_hint_store
from src.agent.strategies.structured import resolve_hints_for_schemas
from src.ingestion.tabular_store import connect_tabular, schema_prompt_with_values
async def main():
    ms = get_metadata_store(); hs = get_hint_store()
    await populate_hint_store(ms, hs)
    schemas = [s for s in await ms.load_all_schemas() if 'all_gs' in s.table]
    hints = await resolve_hints_for_schemas(schemas, hs, ms)
    con = connect_tabular(read_only=True)
    p = schema_prompt_with_values(schemas, con, hints=hints)
    print([l for l in p.splitlines() if 'locname' in l.lower()][0][:300])
asyncio.run(main())
"
```

Expected: the `locname` values line now shows `TU (Tampa, FL)`, `RUS (Rest of U.S.)`. Then run the question "What are the GS salary rates in Tampa?" in the playground and confirm a real answer with the source-document citation (verifies this stacks with PR #2's citation work).

- [ ] **Step 3: Update the roadmap memory**

Edit `/home/mike/.claude/projects/-home-mike-sauron/memory/tabular-spreadsheet-ingestion-roadmap.md`: note that extensible table hints (Phase 1, curated) are implemented on branch `feat/extensible-table-hints` — `SchemaHint`/`HintStore` (`src/db/hint_store.py`), persisted in metadata.db (`schema_hints` table) + `MetadataStore` CRUD, loaded at startup via `populate_hint_store`, resolved per-table by `resolve_hints`/`resolve_hints_for_schemas` (owning-doc category/dataset scope, precedence curated>learned>auto), and injected into the text-to-SQL prompt by `schema_prompt_with_values(..., hints=)` (glossary annotates the `values:` line as `CODE (meaning)`). Admin endpoints `/hints` (+ `/hints/bulk`) create/list/delete live+persisted. This is the general fix for the cryptic locality-code gap (TU=Tampa) — Phases 2 (auto via profiler) and 3 (learned via feedback) deferred. Spec/plan `docs/superpowers/{specs,plans}/2026-05-27-extensible-table-hints*`.

- [ ] **Step 4: Commit any remaining repo changes**

```bash
git add -A && git commit -m "docs: note extensible table hints in roadmap memory" || echo "nothing to commit"
```

(The memory file lives under `.claude/`, outside the repo tree; the `|| echo` keeps the step green if nothing in the repo is staged.)

---

## Self-Review (completed during authoring)

**Spec coverage:**
- Data model `SchemaHint` (scope/type/target/payload/provenance/confidence) → Task 1 (dataclass) + Task 2 (ORM).
- Storage `HintStore` + `MetadataStore` CRUD + startup load → Tasks 1, 2, 3.
- Resolver (scope merge, precedence curated>learned>auto, column-name targeting, fail-safe) → Task 4.
- Injection (glossary annotates `values:`, column/table notes, `hints=None` byte-identical) → Task 5.
- Query-time data flow (map table→doc, resolve, pass into prompt) → Task 6.
- Authoring (admin API + bulk import, live+persisted) → Task 7.
- Error handling (fail-open at resolve + inject + list) → Tasks 4, 6 (`try/except`), 7 (skip malformed).
- Testing per spec → each task's tests; regression + e2e → Task 8.

**Placeholder scan:** Task 7 intentionally defers the exact router name + admin-auth dependency to Step 1 inspection (they are existing codebase facts, not undefined types) and marks the substitution explicitly; every other step shows complete code.

**Type consistency:** `SchemaHint(scope_type, scope_value, hint_type, target_column, payload, provenance, confidence, id, created_at, created_by)` defined in Task 1 and constructed identically in Tasks 2/3/4/6/7. `ResolvedHints(column_glossaries, column_notes, table_notes)` defined in Task 4 and consumed in Task 5. `resolve_hints(table_schema, doc_record, hint_store)` (Task 4) called by `resolve_hints_for_schemas` (Task 6). `schema_prompt_with_values(schemas, con, max_distinct=100, hints=None)` (Task 5) called with `hints=` in Task 6. `populate_hint_store(metadata_store, hint_store)` (Task 3) reused in Task 7's `delete_hint_impl`.
