# Structured-Retrieval Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make tabular/structured questions return correct, complete answers regardless of which strategy classifies them, and make the same question classify consistently across runs.

**Architecture:** Phase 1 fixes the synthesizer so the SWEEP branch keeps precise structured `table_row` narratives (dropping only bulky raw sweep chunks) when map-reduce is present. Phase 2 removes run-to-run classification flips via three independent hygiene changes: a fixed LLM seed, stable table-list ordering in the classifier prompt, and a purge of orphaned/test schemas. Each task is self-contained and independently testable.

**Tech Stack:** Python 3, pytest, pydantic-settings, DuckDB. No new dependencies.

Spec: `docs/superpowers/specs/2026-05-27-structured-retrieval-reliability-design.md`.

---

## Task 1: Synthesizer keeps structured narratives in SWEEP (Phase 1)

**Files:**
- Modify: `src/agent/synthesizer.py:108-117`
- Test: `tests/test_agent/test_synthesizer.py`

The bug: when any `map-reduce` chunk is present, `regular_chunks = []` drops every non-synthetic chunk — including the precise structured `table_row` narratives that `retrieve_structured` produced. Fix: keep the `table_row` narratives, drop only the bulky raw sweep chunks (other tiers).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_synthesizer.py`:

```python
def _chunk(text, doc_id="d1", tier="medium", score=0.9):
    return RetrievedChunk(
        text=text, score=score,
        metadata=ChunkMetadata(
            doc_id=doc_id, filename="pay.xlsx", doc_type="xlsx",
            chunk_index=0, start_char=0, acl_groups=["finance"],
            chunk_size_tier=tier,
        ),
    )


def test_sweep_keeps_structured_narratives_drops_raw_when_mapreduce():
    """With a map-reduce synthesis present, structured table_row narratives are
    kept in the synthesis context but bulky raw sweep chunks are dropped."""
    captured = {}

    def fake_generate(**kwargs):
        captured["user_prompt"] = kwargs.get("user_prompt", "")
        return "answer [1]"

    mr = _chunk("Map-reduce synthesis of pay docs.", doc_id="map-reduce", tier="medium")
    narrative = _chunk("locality=Tampa, grade=GS-12: salary is 86415", doc_id="dpay", tier="table_row")
    raw = _chunk("RAWSWEEPBLOB huge raw spreadsheet text", doc_id="dpay", tier="large")

    with patch("src.agent.synthesizer.generate", fake_generate):
        state = AgentState(
            question="GS rates in Tampa", user_groups=["finance"],
            query_type=QueryType.SWEEP,
            retrieved_chunks=[mr, narrative, raw], sql_results=[],
        )
        synthesize_answer(state)

    ctx = captured["user_prompt"]
    assert "Map-reduce synthesis" in ctx          # synthetic kept
    assert "locality=Tampa" in ctx                # structured narrative kept
    assert "RAWSWEEPBLOB" not in ctx              # raw sweep chunk dropped


def test_no_mapreduce_keeps_raw_chunks():
    """Regression: without a map-reduce chunk, raw chunks are still included."""
    captured = {}

    def fake_generate(**kwargs):
        captured["user_prompt"] = kwargs.get("user_prompt", "")
        return "answer [1]"

    raw = _chunk("RAWSWEEPBLOB huge raw spreadsheet text", doc_id="dpay", tier="large")
    with patch("src.agent.synthesizer.generate", fake_generate):
        state = AgentState(
            question="GS rates in Tampa", user_groups=["finance"],
            query_type=QueryType.LOOKUP,
            retrieved_chunks=[raw], sql_results=[],
        )
        synthesize_answer(state)
    assert "RAWSWEEPBLOB" in captured["user_prompt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent/test_synthesizer.py::test_sweep_keeps_structured_narratives_drops_raw_when_mapreduce -v`
Expected: FAIL — `RAWSWEEPBLOB` is absent today only because ALL regular chunks are dropped, so `locality=Tampa` is ALSO absent → the `assert "locality=Tampa" in ctx` fails.

- [ ] **Step 3: Implement the fix**

In `src/agent/synthesizer.py`, replace the `if has_map_reduce:` block (lines ~108-117):

```python
    if has_map_reduce:
        # Map-reduce distilled the prose docs; raw doc chunks are redundant.
        # Structured table_row narratives are precise and compact — keep them so
        # SWEEP never discards retrieved structured data.
        regular_chunks = sorted(
            [c for c in chunks
             if c.metadata.doc_id not in SYNTHETIC_IDS
             and c.metadata.chunk_size_tier == "table_row"],
            key=lambda c: c.score, reverse=True,
        )
        logger.info(
            "Synthesizer: map-reduce synthesis + %d structured narratives "
            "(raw sweep chunks skipped)", len(regular_chunks))
    else:
        regular_chunks = sorted(
            [c for c in chunks if c.metadata.doc_id not in SYNTHETIC_IDS],
            key=lambda c: c.score, reverse=True,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent/test_synthesizer.py -v`
Expected: PASS (new tests + existing synthesizer tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/agent/synthesizer.py tests/test_agent/test_synthesizer.py
git commit -m "fix: SWEEP synthesizer keeps structured table_row narratives, drops only raw chunks"
```

---

## Task 2: Fixed LLM seed for deterministic classification (Phase 2.1)

**Files:**
- Modify: `src/config.py:63` (add setting)
- Modify: `src/generation/llm_client.py:29-34` (add seed to payload)
- Test: `tests/test_generation/test_llm_client.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generation/test_llm_client.py`:

```python
def test_call_llm_includes_seed(monkeypatch):
    """_call_llm sends a deterministic seed in the request payload."""
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, headers, timeout, verify):
        captured["payload"] = json
        return FakeResp()

    monkeypatch.setattr("src.generation.llm_client.requests.post", fake_post)
    from src.generation.llm_client import _call_llm
    _call_llm([{"role": "user", "content": "hi"}], model="m", temperature=0.0, max_tokens=10)
    assert captured["payload"]["seed"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generation/test_llm_client.py::test_call_llm_includes_seed -v`
Expected: FAIL — `KeyError: 'seed'` (payload has no seed yet).

- [ ] **Step 3: Add the config setting**

In `src/config.py`, after the `llm_max_output_tokens` line (line 63), add:

```python
    llm_seed: int = 0  # fixed seed for deterministic LLM sampling (classification stability)
```

- [ ] **Step 4: Add seed to the payload**

In `src/generation/llm_client.py`, change the `payload` dict in `_call_llm` (lines 29-34) to:

```python
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": settings.llm_seed,
    }
```

(`settings` is already imported at the top of the file.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_generation/test_llm_client.py -v`
Expected: PASS (new test + existing llm_client tests unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/config.py src/generation/llm_client.py tests/test_generation/test_llm_client.py
git commit -m "feat: pass fixed llm_seed to LLM calls for deterministic classification"
```

---

## Task 3: Stable table ordering in the classifier prompt (Phase 2.2)

**Files:**
- Modify: `src/agent/classifier.py:22-24`
- Test: `tests/test_agent/test_classifier.py`

`format_available_tables` renders registered tables into the classifier prompt in registry iteration order, which shifts as documents are ingested/removed. Sort by table name so the prompt text is identical run-to-run.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_classifier.py`:

```python
from types import SimpleNamespace
from src.agent.classifier import format_available_tables


def test_format_available_tables_is_order_stable():
    a = SimpleNamespace(table="doc_a_pay", description="A pay")
    b = SimpleNamespace(table="doc_b_pay", description="B pay")
    c = SimpleNamespace(table="doc_c_pay", description="C pay")
    out1 = format_available_tables([c, a, b])
    out2 = format_available_tables([b, c, a])
    assert out1 == out2
    assert out1 == "- doc_a_pay: A pay\n- doc_b_pay: B pay\n- doc_c_pay: C pay"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent/test_classifier.py::test_format_available_tables_is_order_stable -v`
Expected: FAIL — current output preserves input order, so `out1 != out2`.

- [ ] **Step 3: Sort the schemas**

In `src/agent/classifier.py`, change `format_available_tables` (lines 22-24) to:

```python
def format_available_tables(schemas) -> str:
    """One '- <table>: <description>' line per schema, sorted by table name for
    a stable (run-to-run identical) classifier prompt."""
    return "\n".join(
        f"- {s.table}: {s.description}" for s in sorted(schemas, key=lambda s: s.table)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent/test_classifier.py -v`
Expected: PASS (new test + existing classifier tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/agent/classifier.py tests/test_agent/test_classifier.py
git commit -m "feat: sort registered tables in classifier prompt for stable classification"
```

---

## Task 4: Purge orphaned/test schemas (Phase 2.3)

**Files:**
- Modify: `src/ingestion/tabular_ingest.py` (add `purge_orphan_schemas`)
- Test: `tests/test_ingestion/test_tabular_ingest.py`

Add a function that drops DuckDB tables and deletes/unregisters schemas whose `doc_id` has no live document in the metadata store. This removes leftover test fixtures (toy `Tampa/Boston/Denver`, `doc1_*`) by orphan status — not hardcoded names — so a polluted, shifting table list stops flipping the classifier. Modeled on the existing `cleanup_spreadsheet_tables` (same store-access pattern, fail-open). All tabular tables are namespaced `doc_<safe_doc_id>_<sheet>` via `duckdb_table_name`, so only `doc_`-prefixed objects are ever considered.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ingestion/test_tabular_ingest.py`:

```python
from types import SimpleNamespace


@pytest.mark.asyncio
async def test_purge_orphan_schemas_removes_only_orphans(monkeypatch):
    from src.ingestion.tabular_ingest import purge_orphan_schemas
    from src.ingestion.tabular_store import duckdb_table_name

    live_tbl = duckdb_table_name("live1", "pay")
    ghost_tbl = duckdb_table_name("ghost", "pay")

    class FakeMS:
        def __init__(self):
            self.deleted = []

        async def list_documents(self, user_groups=None):
            return [SimpleNamespace(doc_id="live1", acl_groups=["ALL"])]

        async def load_all_schemas(self):
            return [
                SimpleNamespace(database="spreadsheets", table=live_tbl),
                SimpleNamespace(database="spreadsheets", table=ghost_tbl),
            ]

        async def delete_schema(self, database, table):
            self.deleted.append((database, table))

    class FakeReg:
        def __init__(self):
            self.removed = []

        def remove(self, database, table):
            self.removed.append((database, table))

    class FakeCon:
        def __init__(self, tables):
            self.tables = tables
            self.dropped = []

        def execute(self, sql):
            if sql.startswith("SELECT table_name"):
                rows = [(t,) for t in self.tables]
                return SimpleNamespace(fetchall=lambda: rows)
            if sql.startswith("DROP TABLE"):
                self.dropped.append(sql)
            return SimpleNamespace(fetchall=lambda: [])

        def close(self):
            pass

    fake_con = FakeCon([live_tbl, ghost_tbl, "system_meta"])
    monkeypatch.setattr("src.ingestion.tabular_store.connect_tabular", lambda read_only=False: fake_con)

    ms, reg = FakeMS(), FakeReg()
    removed = await purge_orphan_schemas(ms, schema_registry=reg)

    assert removed == 1
    assert ms.deleted == [("spreadsheets", ghost_tbl)]
    assert reg.removed == [("spreadsheets", ghost_tbl)]
    assert any(ghost_tbl in d for d in fake_con.dropped)        # orphan table dropped
    assert not any(live_tbl in d for d in fake_con.dropped)     # live table kept
    assert not any("system_meta" in d for d in fake_con.dropped)  # non-doc table untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestion/test_tabular_ingest.py::test_purge_orphan_schemas_removes_only_orphans -v`
Expected: FAIL — `ImportError: cannot import name 'purge_orphan_schemas'`.

- [ ] **Step 3: Implement `purge_orphan_schemas`**

In `src/ingestion/tabular_ingest.py`, add after `cleanup_spreadsheet_tables` (after line 231):

```python
async def purge_orphan_schemas(metadata_store, schema_registry=None) -> int:
    """Drop DuckDB tables and delete/unregister schemas whose doc_id has no live
    document in the metadata store (e.g. leftover test fixtures). Only operates on
    the ``doc_`` table namespace. Fail-open. Returns the number of orphan schemas
    removed."""
    if schema_registry is None:
        from src.api.routes_ingest import get_schema_registry
        schema_registry = get_schema_registry()

    from src.ingestion.tabular_store import connect_tabular, duckdb_table_name

    live_docs = await metadata_store.list_documents()
    live_prefixes = [duckdb_table_name(d.doc_id, "") for d in live_docs]

    def _is_orphan(table_name: str) -> bool:
        # Only our namespace; orphan if no live doc prefix matches.
        return table_name.startswith("doc_") and not any(
            table_name.startswith(p) for p in live_prefixes)

    removed = 0
    try:
        for sc in await metadata_store.load_all_schemas():
            if _is_orphan(sc.table):
                await metadata_store.delete_schema(sc.database, sc.table)
                schema_registry.remove(sc.database, sc.table)
                removed += 1
    except Exception as e:
        logger.warning(f"Orphan schema purge failed: {e}")

    try:
        con = connect_tabular(read_only=False)
        try:
            tables = [r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables").fetchall()]
            for t in tables:
                if _is_orphan(t):
                    con.execute(f'DROP TABLE IF EXISTS "{t}"')
        finally:
            con.close()
    except Exception as e:
        logger.warning(f"Orphan DuckDB table purge failed: {e}")

    logger.info(f"Orphan schema purge removed {removed} schema(s)")
    return removed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingestion/test_tabular_ingest.py -v`
Expected: PASS (new test + existing tabular_ingest tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_ingest.py tests/test_ingestion/test_tabular_ingest.py
git commit -m "feat: purge_orphan_schemas drops schemas/tables with no live document"
```

---

## Task 5: Full regression + manual verification

**Files:** none (verification only)

- [ ] **Step 1: Run the affected test suites**

Run: `pytest tests/test_agent/ tests/test_generation/ tests/test_ingestion/ -q`
Expected: the new tests pass and no NEW failures appear. Pre-existing/environmental failures are known and unrelated: `tests/test_ingestion` carries 7 (numpy/sklearn binary `numpy.dtype size changed`, OpenAI 401, stale `embedder.OpenAI`/`pipeline.extract_entities` patches) and `tests/test_agent` carries ~8 (test_graph/lookup/sweep/cross_reference harness/mocking). Confirm the failing set is unchanged from before this branch (e.g. via `git stash` + rerun if unsure).

- [ ] **Step 2: Run the orphan purge against the live store (operator step)**

This is the one-time cleanup that removes the toy fixtures. Run inside the API container so it uses the deployed data volume:

```bash
docker compose exec -T api python - <<'PY'
import asyncio
from src.ingestion.tabular_ingest import purge_orphan_schemas
from src.api.routes_ingest import get_metadata_store, get_schema_registry
async def main():
    ms = get_metadata_store()
    n = await purge_orphan_schemas(ms, schema_registry=get_schema_registry())
    print("orphan schemas removed:", n)
asyncio.run(main())
PY
```

Expected: prints the count of removed orphan schemas (the toy `Tampa/Boston/Denver` and `doc1_*` tables, if present in the deployed store).

- [ ] **Step 3: Manual smoke — the Tampa question is stable and answered**

In the playground, run "What are the GS salary rates in Tampa?" two or three times. Tail the logs:

```bash
docker compose logs --since 5m api | grep -E "Classified|Text-to-SQL|Synthesizer"
```

Expected: the `Classified ... -> <type>` line is the SAME type across repeats (determinism), and the answer contains the GS rows whether the type is `sweep` or `analytical` (if `sweep`, the `Synthesizer: map-reduce synthesis + N structured narratives` line shows N > 0).

- [ ] **Step 4: Update the roadmap memory**

Edit `/home/mike/.claude/projects/-home-mike-sauron/memory/tabular-spreadsheet-ingestion-roadmap.md`: note that the SWEEP synthesizer no longer discards structured `table_row` narratives, that classification determinism hygiene (fixed seed, sorted table prompt, orphan purge) is in, and remove the now-resolved "structured-retrieval reliability" concern. Keep the still-open deferred items (the ≥0.30 structured gate in sweep; wiring `strategy_memory.get_best_strategy`; query-cache `query_type` reuse).

---

## Self-Review (completed during authoring)

**Spec coverage:**
- Phase 1 (synthesizer keeps structured narratives, drops raw) → Task 1.
- Phase 2.1 (fixed LLM seed) → Task 2.
- Phase 2.2 (stable table ordering) → Task 3.
- Phase 2.3 (purge orphaned/test schemas) → Task 4.
- Verification (regression + the Tampa smoke) → Task 5.
- Out-of-scope items from the spec (≥0.30 gate, strategy_memory wiring, cache query_type reuse, locality robustness) are intentionally NOT tasked; carried into the memory note in Task 5 Step 4.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; commands have expected output.

**Type consistency:** `format_available_tables` consumes objects with `.table`/`.description` (Task 3 test uses `SimpleNamespace` with both; production `TableSchema` has both). `purge_orphan_schemas(metadata_store, schema_registry=None)` matches the call in Task 5 Step 2 and the test in Task 4. `duckdb_table_name(doc_id, sheet)` used consistently for prefixes. `chunk_size_tier` is the existing `ChunkMetadata` field (default "medium"); `"table_row"` is the existing tier emitted by `build_row_narratives`/`messy_region_narratives`. `settings.llm_seed` defined in Task 2 Step 3 and read in Task 2 Step 4.
