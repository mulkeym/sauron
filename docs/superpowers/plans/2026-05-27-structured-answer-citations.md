# Structured/SQL Answer Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit a `Citation` to the source document of each table the executed SQL referenced, so structured/SQL answers (especially ANALYTICAL, which returns no chunks) carry provenance.

**Architecture:** A pure `referenced_source_docs(sql, live_doc_ids)` resolver in `tabular_store.py` maps an executed SQL statement to the owning document ids via the existing `_referenced_tables` parse + the `duckdb_table_name(doc_id,"")` prefix match. The synthesizer, when `state["structured_trace"]` shows a lookup that ran with rows, resolves those docs through the metadata store and appends one `Citation` per source doc, deduped by `doc_id` against the existing chunk-derived citations. Fully additive and fail-open.

**Tech Stack:** Python 3, pytest, DuckDB, pydantic. No new dependencies.

Spec: `docs/superpowers/specs/2026-05-27-structured-answer-citations-design.md`.

---

## Task 1: `referenced_source_docs` pure resolver

**Files:**
- Modify: `src/ingestion/tabular_store.py` (add function after `duckdb_table_name`)
- Test: `tests/test_ingestion/test_tabular_store.py`

`_referenced_tables(sql)` (returns lowercased table names) and `_cte_names(sql)` and `duckdb_table_name(doc_id, sheet)` (returns lowercase `doc_<safe_doc>_<safe_sheet>`) already exist in this file.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ingestion/test_tabular_store.py`:

```python
from src.ingestion.tabular_store import referenced_source_docs, duckdb_table_name


def test_referenced_source_docs_single():
    sql = f'SELECT * FROM "{duckdb_table_name("live1", "pay")}" WHERE x = 1'
    assert referenced_source_docs(sql, ["live1", "other"]) == ["live1"]


def test_referenced_source_docs_join_two_order_stable():
    t1 = duckdb_table_name("a", "s")
    t2 = duckdb_table_name("b", "s")
    sql = f'SELECT * FROM "{t1}" JOIN "{t2}" ON 1=1'
    assert referenced_source_docs(sql, ["a", "b"]) == ["a", "b"]


def test_referenced_source_docs_skips_unmatched():
    sql = f'SELECT * FROM "{duckdb_table_name("ghost", "pay")}"'
    assert referenced_source_docs(sql, ["live1"]) == []


def test_referenced_source_docs_ignores_cte():
    t = duckdb_table_name("live1", "pay")
    sql = f'WITH tmp AS (SELECT 1) SELECT * FROM "{t}", tmp'
    assert referenced_source_docs(sql, ["live1"]) == ["live1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_tabular_store.py -k referenced_source_docs -v`
Expected: FAIL — `ImportError: cannot import name 'referenced_source_docs'`.

- [ ] **Step 3: Implement the resolver**

In `src/ingestion/tabular_store.py`, add immediately AFTER the `duckdb_table_name` function:

```python
def referenced_source_docs(sql: str, live_doc_ids: list[str]) -> list[str]:
    """Source document ids whose DuckDB tables the SQL references.

    Parses table names via ``_referenced_tables`` (minus CTE aliases) and matches
    each against the per-doc table-name prefix ``duckdb_table_name(doc_id, "")``.
    Order-stable by ``live_doc_ids``; deduped; referenced tables with no live doc
    (e.g. since-deleted docs, or non-doc names) are skipped. Pure — no I/O."""
    tables = _referenced_tables(sql) - _cte_names(sql)   # both already lowercased
    out: list[str] = []
    for doc_id in live_doc_ids:
        prefix = duckdb_table_name(doc_id, "")
        if doc_id not in out and any(t.startswith(prefix) for t in tables):
            out.append(doc_id)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion/test_tabular_store.py -v`
Expected: PASS (4 new tests + all pre-existing tabular_store tests).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_store.py tests/test_ingestion/test_tabular_store.py
git commit -m "feat: referenced_source_docs — map executed SQL to its source document ids"
```

End the commit message with:
```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 2: Append SQL-source citations in the synthesizer

**Files:**
- Modify: `src/agent/synthesizer.py` (after the citations list, before `return`, ~line 208)
- Test: `tests/test_agent/test_synthesizer.py`

`synthesize_answer` builds `citations` from `chunks` (line 196-208) then `return {"answer": answer, "citations": citations}` (line 209). `Citation` fields: `doc_id, filename, doc_type, chunk_index, page, snippet, relevance, source_url`. `state` carries `structured_trace` (`{sql, status, row_count, ...}`). The test file already defines `_chunk(text, doc_id="d1", tier="medium", score=0.9)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_synthesizer.py`:

```python
def _sql_doc_ms(doc_id="docpay", filename="2026-pay.xlsx"):
    class _Doc:
        def __init__(self):
            self.doc_id = doc_id
            self.filename = filename
            self.doc_type = "xlsx"
            self.source_url = ""

    class _MS:
        async def list_documents(self, user_groups=None):
            return [_Doc()]

        async def get_document(self, did):
            return _Doc() if did == doc_id else None

    return _MS()


def test_sql_answer_cites_source_document(monkeypatch):
    from src.ingestion.tabular_store import duckdb_table_name
    tbl = duckdb_table_name("docpay", "pay")
    monkeypatch.setattr("src.api.routes_ingest.get_metadata_store", lambda: _sql_doc_ms())
    with patch("src.agent.synthesizer.generate", lambda **k: "answer"):
        state = AgentState(
            question="pay?", user_groups=["finance"], query_type=QueryType.ANALYTICAL,
            retrieved_chunks=[], sql_results=[{"salary": 91162}],
            structured_trace={"status": "ran", "row_count": 15, "sql": f'SELECT * FROM "{tbl}"'},
        )
        result = synthesize_answer(state)
    cits = [c for c in result["citations"] if c.doc_id == "docpay"]
    assert len(cits) == 1
    assert cits[0].filename == "2026-pay.xlsx"
    assert "15 rows" in cits[0].snippet
    assert cits[0].relevance == 1.0


def test_sql_citation_deduped_with_chunk(monkeypatch):
    from src.ingestion.tabular_store import duckdb_table_name
    tbl = duckdb_table_name("docpay", "pay")
    monkeypatch.setattr("src.api.routes_ingest.get_metadata_store", lambda: _sql_doc_ms())
    chunk = _chunk("locality=Tampa: salary 91162", doc_id="docpay", tier="table_row")
    with patch("src.agent.synthesizer.generate", lambda **k: "answer"):
        state = AgentState(
            question="pay?", user_groups=["finance"], query_type=QueryType.SWEEP,
            retrieved_chunks=[chunk], sql_results=[{"salary": 91162}],
            structured_trace={"status": "ran", "row_count": 15, "sql": f'SELECT * FROM "{tbl}"'},
        )
        result = synthesize_answer(state)
    assert len([c for c in result["citations"] if c.doc_id == "docpay"]) == 1


def test_no_sql_citation_when_zero_rows(monkeypatch):
    from src.ingestion.tabular_store import duckdb_table_name
    tbl = duckdb_table_name("docpay", "pay")
    monkeypatch.setattr("src.api.routes_ingest.get_metadata_store", lambda: _sql_doc_ms())
    chunk = _chunk("some prose", doc_id="d1", tier="large")
    with patch("src.agent.synthesizer.generate", lambda **k: "answer"):
        state = AgentState(
            question="pay?", user_groups=["finance"], query_type=QueryType.ANALYTICAL,
            retrieved_chunks=[chunk], sql_results=[],
            structured_trace={"status": "ran", "row_count": 0, "sql": f'SELECT * FROM "{tbl}"'},
        )
        result = synthesize_answer(state)
    assert all(c.doc_id != "docpay" for c in result["citations"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_synthesizer.py::test_sql_answer_cites_source_document -v`
Expected: FAIL — `cits` is empty (no SQL citation is produced today; ANALYTICAL has no chunks).

- [ ] **Step 3: Implement the SQL-citation block**

In `src/agent/synthesizer.py`, REPLACE the final `return {"answer": answer, "citations": citations}` (line ~209) with:

```python
    # Structured/SQL answers: cite the source document(s) of the table(s) the
    # executed SQL referenced. ANALYTICAL returns no chunks, so without this the
    # answer would carry zero citations. Fully additive + fail-open.
    trace = state.get("structured_trace") or {}
    if trace.get("status") == "ran" and trace.get("row_count", 0) > 0 and trace.get("sql"):
        try:
            import asyncio
            from src.api.routes_ingest import get_metadata_store
            from src.ingestion.tabular_store import referenced_source_docs
            _ms = get_metadata_store()

            async def _fetch_sql_docs():
                docs = await _ms.list_documents()
                src_ids = referenced_source_docs(trace["sql"], [d.doc_id for d in docs])
                by_id = {d.doc_id: d for d in docs}
                return [by_id[i] for i in src_ids if i in by_id]

            try:
                sql_docs = asyncio.run(_fetch_sql_docs())
            except RuntimeError:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    sql_docs = pool.submit(asyncio.run, _fetch_sql_docs()).result()

            existing = {c.doc_id for c in citations}
            for rec in sql_docs:
                if rec.doc_id in existing:
                    continue
                citations.append(Citation(
                    doc_id=rec.doc_id,
                    filename=rec.filename,
                    doc_type=getattr(rec, "doc_type", "") or "",
                    chunk_index=0,
                    page=None,
                    snippet=f"Structured query returned {trace['row_count']} rows from this table.",
                    relevance=1.0,
                    source_url=getattr(rec, "source_url", "") or "",
                ))
                existing.add(rec.doc_id)
        except Exception as e:
            logger.debug(f"SQL-source citations skipped: {e}")

    return {"answer": answer, "citations": citations}
```

(`Citation` and `logger` are already imported/defined in this module; `asyncio`/`concurrent.futures` are imported locally to mirror the existing `source_url` lookup block.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent/test_synthesizer.py -v`
Expected: PASS — the 3 new tests plus all pre-existing synthesizer tests (the new block only runs when the trigger holds, so other tests are unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/agent/synthesizer.py tests/test_agent/test_synthesizer.py
git commit -m "feat: cite source document(s) of tables referenced by structured/SQL answers"
```

End with the `Co-Authored-By` trailer.

---

## Task 3: Regression + manual smoke + memory

**Files:** none (verification); then memory update

- [ ] **Step 1: Run the affected suites**

Run: `python -m pytest tests/test_ingestion/ tests/test_agent/test_synthesizer.py -q`
Expected: new tests pass; no NEW failures vs. before this branch. (`tests/test_ingestion` carries the known ~7 pre-existing environmental failures — numpy/sklearn ABI, OpenAI 401, stale patches; confirm the failing set is unchanged, e.g. via `git stash` + rerun if unsure. The synthesizer tests must all PASS.)

- [ ] **Step 2: Manual smoke in the deployed playground**

Rebuild + redeploy, then verify an ANALYTICAL answer now carries a citation:

```bash
docker compose up -d --build api
```

In the playground, run "What is the GS-12 salary in Tampa?" (a question that classifies analytical / runs SQL). Confirm the answer's **Citations** now include the source spreadsheet (e.g. `2026-general-schedule-pay-rates.xlsx`) with the snippet "Structured query returned N rows from this table." Then run a question that returns 0 SQL rows or a prose-only question and confirm no spurious SQL citation appears.

- [ ] **Step 3: Update the roadmap memory**

Edit `/home/mike/.claude/projects/-home-mike-sauron/memory/tabular-spreadsheet-ingestion-roadmap.md`: note that structured/SQL answers now cite the source document(s) of the table(s) the executed SQL referenced, via the pure `referenced_source_docs(sql, live_doc_ids)` resolver in `tabular_store.py` (table-name prefix match, same scheme as `cleanup_spreadsheet_tables`) + a fail-open citation block in `synthesize_answer` (triggers on `structured_trace.status=="ran"` and `row_count>0`, deduped by doc_id, snippet "Structured query returned N rows from this table", relevance 1.0). Note this also surfaces WHICH table/doc answered — useful against the still-open multi-table/year disambiguation gap.

- [ ] **Step 4: Commit any remaining repo changes**

```bash
git add -A && git commit -m "docs: note structured-answer citations in roadmap memory" || echo "nothing to commit"
```

(The memory file lives under `.claude/`, outside the repo tree; the `|| echo` keeps the step green if nothing in the repo is staged.)

---

## Self-Review (completed during authoring)

**Spec coverage:**
- Trigger (`status=="ran"` and `row_count>0`) → Task 2 block condition.
- `referenced_source_docs` pure resolver (parse via `_referenced_tables`/`_cte_names`, prefix match, dedup, order-stable, skip unmatched) → Task 1.
- Synthesizer citation construction (one per source doc, `chunk_index=0`, `page=None`, summary snippet, `relevance=1.0`, `source_url` carried) → Task 2.
- Dedup/merge by `doc_id` against chunk citations → Task 2 (`existing` set).
- Fail-open (wrapped, `logger.debug` on failure) → Task 2.
- Testing: resolver cases (single/join/skip/CTE) → Task 1; synthesizer cases (analytical cite, dedup, zero-row none) → Task 2.

**Placeholder scan:** none — every code step shows complete code; commands have expected output.

**Type consistency:** `referenced_source_docs(sql: str, live_doc_ids: list[str]) -> list[str]` is defined in Task 1 and called identically in Task 2. `Citation(...)` keyword args match the model fields confirmed in the spec (`doc_id, filename, doc_type, chunk_index, page, snippet, relevance, source_url`). `structured_trace` keys (`status`, `row_count`, `sql`) match what the Structured Lookup feature put on `AgentState`. `duckdb_table_name` / `_referenced_tables` / `_cte_names` are the real existing names in `tabular_store.py`.
