# File-metadata (catalog) Q&A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer questions about the document catalog (counts, per-file facts, content-tag search, overview) via a new `METADATA` query type that runs text-to-SQL over an ACL-pre-filtered, in-memory DuckDB catalog.

**Architecture:** The classifier routes catalog questions to `METADATA`. A new strategy `retrieve_metadata_catalog` calls `MetadataStore.list_documents(user_groups)` (ACL-filtered), loads those rows into an ephemeral in-memory DuckDB `files` table (plus `datasets`/`categories`), runs text-to-SQL against it (reusing existing `generate_sql`/`execute_duckdb_sql`), returns rows + file citations, and falls back to a catalog snapshot on error. ACL is enforced in Python before any SQL runs.

**Tech Stack:** Python, DuckDB (`:memory:`, `enable_external_access=False`), pytest (`pytest-asyncio`), LangGraph. Spec: `docs/superpowers/specs/2026-05-29-file-metadata-catalog-qa-design.md`.

**Reference facts (verified in current code):**
- `QueryType(StrEnum)` in `src/agent/state.py:19` — values lookup/sweep/analytical/cross_reference/temporal. `classify_query` does `QueryType(parsed["query_type"])`.
- `MetadataStore.list_documents(user_groups=None)` (`src/db/metadata.py:113`) returns `DocumentRecord`s; when `user_groups` given, filters to docs whose `acl_groups` overlap. `list_datasets(active_only=True)` returns `Dataset`s (`.id`, `.name`, `.description`). `list_categories()` returns `Category`s (`.name`, `.description`).
- `DocumentRecord` fields: `doc_id, filename, doc_type, content_hash, dataset_id, category, acl_groups, chunk_count, source_url, summary, metadata_tags(dict), uploaded_by, created_at`.
- `execute_duckdb_sql(con, sql, allowed_tables=None)` (`src/ingestion/tabular_store.py:165`) — single read-only SELECT/WITH; raises on violations; `allowed_tables` is the table allow-list. Returns `list[dict]`.
- `generate_sql(schema_prompt, question, generate_fn=None)` (`src/agent/strategies/structured.py:107`) — LLM text-to-SQL; `generate_fn` is injectable for tests.
- `StructuredLookupTrace` (`src/agent/strategies/structured.py:76`) — `query_type, sql, status('ran'|'skipped'|'error'), error, row_count, sample_rows, fell_back, rows`, `.to_dict()`.
- `RetrievedChunk` / `ChunkMetadata` (`src/retrieval/models.py:5,19`): chunk = `{text, score, metadata}`; metadata = `{doc_id, filename, doc_type, chunk_index, start_char, acl_groups, ...}`.
- Graph retrieve node (`src/agent/graph.py:136-143`) is an `elif query_type == ...` chain; the closure captures `metadata_store`. `SYNTHETIC_IDS` (graph.py:87) already includes `"metadata-context"` (score-exempt synthetic chunk).
- Strategy-Memory guard added on the prior branch (`src/agent/classifier.py`): inside the override `elif`, `if llm_pick == QueryType.ANALYTICAL:` suppresses the override (reason "protected").
- DuckDB in-memory connection: `duckdb.connect(":memory:", config={"enable_external_access": False})` (same guard `connect_tabular` uses).

**Running tests during implementation:** host Python is too old; the baked image lacks new files. Mount host `src` + `tests` into the image:
`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest <args>`

---

## File Structure

- Create `src/agent/strategies/metadata_catalog.py` — catalog build + text-to-SQL + citations + fallback. One responsibility: catalog Q&A.
- Modify `src/agent/state.py` — add `METADATA` enum value.
- Modify `src/agent/classifier.py` — prompt guidance + extend the guard to METADATA.
- Modify `src/agent/graph.py` — one routing branch.
- Tests: `tests/test_agent/test_metadata_catalog.py` (new), additions to `tests/test_agent/test_classifier.py`.

---

## Task 1: `METADATA` query type + classifier routing + guard

**Files:**
- Modify: `src/agent/state.py`, `src/agent/classifier.py`
- Test: `tests/test_agent/test_classifier.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent/test_classifier.py`:

```python
def test_classify_metadata_routes_when_about_files():
    with patch("src.agent.classifier.generate",
               return_value='{"query_type": "metadata", "sub_tasks": []}'):
        result = classify_query({"question": "How many PDFs do we have?"})
    assert result["query_type"] == QueryType.METADATA


@pytest.mark.asyncio
async def test_memory_does_not_override_metadata(monkeypatch):
    # METADATA is a deterministic capability pick; a learned LOOKUP must not override it.
    monkeypatch.setattr(classifier.settings, "strategy_memory_enabled", True)
    def fake_generate(system_prompt, user_prompt, **kwargs):
        return '{"query_type": "metadata", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)
    async def _best(q):
        return {"strategy": "lookup", "count": 3, "margin": 1.0}
    monkeypatch.setattr(classifier, "get_best_strategy", _best, raising=False)

    node = _classify_node_factory(None)
    out = await node({"question": "how many files are there?", "user_groups": ["ALL"]})

    assert out["query_type"] == QueryType.METADATA
    assert out["strategy_memory"]["overrode"] is False
    assert out["strategy_memory"]["reason"] == "protected"
```

- [ ] **Step 2: Run to verify they fail**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest tests/test_agent/test_classifier.py -k "metadata" -v`
Expected: FAIL — `ValueError: 'metadata' is not a valid QueryType` (enum missing).

- [ ] **Step 3: Add the enum value**

In `src/agent/state.py`, add to `QueryType`:

```python
class QueryType(StrEnum):
    LOOKUP = "lookup"
    SWEEP = "sweep"
    ANALYTICAL = "analytical"
    CROSS_REFERENCE = "cross_reference"
    TEMPORAL = "temporal"
    METADATA = "metadata"
```

- [ ] **Step 4: Add classifier prompt guidance + extend the guard**

In `src/agent/classifier.py`, in `CLASSIFICATION_PROMPT`, add this line to the "Query types:" list (after the `temporal:` line):

```python
- metadata: Question ABOUT the documents/files themselves (the catalog) rather than their content — counts, lists, upload dates, datasets, categories, who uploaded, or which files mention a term. Example: "How many PDFs do we have?", "When was the pay doc uploaded?", "Which files mention officers?", "What datasets exist?", "List files uploaded in May".
```

And add this to the `IMPORTANT:` block:

```python
- Use METADATA only for questions ABOUT the files (catalog: counts, dates, datasets, filenames, which-files-mention). A question answered by the CONTENT of a file (e.g. "what does the pay doc SAY about officers?", "what is the pay for an O-4?") is NOT metadata — use lookup/analytical.
```

In `classify_node`'s override branch, change the guard condition from:

```python
                        if llm_pick == QueryType.ANALYTICAL:
```

to:

```python
                        if llm_pick in (QueryType.ANALYTICAL, QueryType.METADATA):
```

(and update that branch's comment to read "ANALYTICAL/METADATA are capability-gated picks ...").

- [ ] **Step 5: Run to verify pass**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest tests/test_agent/test_classifier.py -v`
Expected: PASS — all classifier tests including the two new ones.

- [ ] **Step 6: Commit**

```bash
git add src/agent/state.py src/agent/classifier.py tests/test_agent/test_classifier.py
git commit -m "feat: add METADATA query type, classifier routing, and guard"
```

---

## Task 2: Catalog-build helpers

**Files:**
- Create: `src/agent/strategies/metadata_catalog.py`
- Test: `tests/test_agent/test_metadata_catalog.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent/test_metadata_catalog.py`:

```python
"""Catalog-build helpers for file-metadata Q&A."""
from types import SimpleNamespace
from datetime import datetime, timezone

from src.agent.strategies.metadata_catalog import (
    _flatten_tags, build_catalog_connection, CATALOG_SCHEMA,
)
from src.ingestion.tabular_store import execute_duckdb_sql


def _doc(doc_id="d1", filename="pay.pdf", doc_type="pdf", dataset_id=2,
         category="payroll", tags=None):
    return SimpleNamespace(
        doc_id=doc_id, filename=filename, doc_type=doc_type, dataset_id=dataset_id,
        category=category, acl_groups=["executives"], chunk_count=5,
        source_url="", summary="active duty pay", uploaded_by="mike",
        created_at=datetime(2025, 5, 1, tzinfo=timezone.utc),
        metadata_tags=tags or {"organizations": ["DoD"], "topics": ["Officer pay"]},
    )


def test_flatten_tags_lowercases_and_joins():
    out = _flatten_tags({"organizations": ["DoD"], "topics": ["Officer Pay"], "amounts": ["$5,000"]})
    assert "dod" in out and "officer pay" in out and "$5,000" in out


def test_flatten_tags_empty():
    assert _flatten_tags(None) == ""
    assert _flatten_tags({}) == ""


def test_build_catalog_has_files_table_with_columns():
    con = build_catalog_connection([_doc()], dataset_names={2: "DoD OPM Policies"})
    rows = execute_duckdb_sql(con, "SELECT doc_id, filename, doc_type, dataset, category, "
                                   "uploaded_by, chunk_count, summary, tags FROM files",
                              allowed_tables={"files"})
    assert len(rows) == 1
    r = rows[0]
    assert r["filename"] == "pay.pdf"
    assert r["dataset"] == "DoD OPM Policies"   # resolved from dataset_id
    assert "dod" in r["tags"]                    # flattened metadata_tags


def test_build_catalog_count_aggregate():
    docs = [_doc(doc_id="a", doc_type="pdf"), _doc(doc_id="b", doc_type="pdf"),
            _doc(doc_id="c", doc_type="xlsx")]
    con = build_catalog_connection(docs, dataset_names={2: "DoD"})
    rows = execute_duckdb_sql(con, "SELECT doc_type, COUNT(*) AS n FROM files GROUP BY doc_type ORDER BY doc_type",
                              allowed_tables={"files"})
    assert rows == [{"doc_type": "pdf", "n": 2}, {"doc_type": "xlsx", "n": 1}]


def test_build_catalog_tag_search_ilike():
    docs = [_doc(doc_id="a", tags={"topics": ["Officer pay"]}),
            _doc(doc_id="b", tags={"topics": ["Enlisted pay"]})]
    con = build_catalog_connection(docs, dataset_names={2: "DoD"})
    rows = execute_duckdb_sql(con, "SELECT doc_id FROM files WHERE tags ILIKE '%officer%'",
                              allowed_tables={"files"})
    assert rows == [{"doc_id": "a"}]


def test_build_catalog_optional_datasets_categories():
    con = build_catalog_connection(
        [_doc()], dataset_names={2: "DoD"},
        datasets=[SimpleNamespace(name="DoD", description="policies")],
        categories=[SimpleNamespace(name="payroll", description="pay records")])
    ds = execute_duckdb_sql(con, "SELECT name FROM datasets", allowed_tables={"datasets"})
    cat = execute_duckdb_sql(con, "SELECT name FROM categories", allowed_tables={"categories"})
    assert ds == [{"name": "DoD"}] and cat == [{"name": "payroll"}]


def test_catalog_schema_mentions_files_and_tags():
    assert "files" in CATALOG_SCHEMA and "tags" in CATALOG_SCHEMA
```

- [ ] **Step 2: Run to verify they fail**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest tests/test_agent/test_metadata_catalog.py -v`
Expected: FAIL — `ModuleNotFoundError: src.agent.strategies.metadata_catalog`.

- [ ] **Step 3: Create the helpers**

Create `src/agent/strategies/metadata_catalog.py`:

```python
"""File-metadata (catalog) Q&A: text-to-SQL over an ACL-pre-filtered in-memory
DuckDB catalog of the documents the asking user can access."""
import logging

logger = logging.getLogger(__name__)

CATALOG_SCHEMA = """Table "files" — one row per document the user can access:
  doc_id VARCHAR, filename VARCHAR, doc_type VARCHAR (e.g. 'pdf','xlsx','docx'),
  dataset VARCHAR (dataset name), category VARCHAR, uploaded_by VARCHAR,
  created_at TIMESTAMP (when the file was uploaded), chunk_count INTEGER,
  summary VARCHAR, tags VARCHAR (lowercased extracted entities/organizations/amounts/topics/identifiers).
Table "datasets" — name VARCHAR, description VARCHAR.
Table "categories" — name VARCHAR, description VARCHAR.
Rules: use ILIKE for case-insensitive text/tag matching; use COUNT for "how many";
when listing specific files, always SELECT filename AND doc_id."""


def _flatten_tags(metadata_tags) -> str:
    """Flatten the extracted metadata_tags dict into one lowercased searchable string."""
    if not metadata_tags:
        return ""
    vals = []
    for field in ("entities", "organizations", "amounts", "topics", "identifiers"):
        for v in (metadata_tags.get(field) or []):
            vals.append(str(v))
    return " ".join(vals).lower()


def build_catalog_connection(docs, dataset_names=None, datasets=None, categories=None):
    """Build an ephemeral in-memory DuckDB with a `files` table (one row per doc)
    and optional `datasets`/`categories` tables. enable_external_access=False so
    LLM-generated SELECTs cannot reach the filesystem or network."""
    import duckdb
    dataset_names = dataset_names or {}
    con = duckdb.connect(":memory:", config={"enable_external_access": False})
    con.execute("""CREATE TABLE files (
        doc_id VARCHAR, filename VARCHAR, doc_type VARCHAR, dataset VARCHAR,
        category VARCHAR, uploaded_by VARCHAR, created_at TIMESTAMP,
        chunk_count INTEGER, summary VARCHAR, tags VARCHAR)""")
    for d in docs:
        con.execute(
            "INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,?)",
            [d.doc_id, d.filename, d.doc_type,
             dataset_names.get(getattr(d, "dataset_id", 0), ""),
             getattr(d, "category", "") or "",
             getattr(d, "uploaded_by", "") or "",
             getattr(d, "created_at", None),
             getattr(d, "chunk_count", 0) or 0,
             getattr(d, "summary", "") or "",
             _flatten_tags(getattr(d, "metadata_tags", None))])
    if datasets is not None:
        con.execute("CREATE TABLE datasets (name VARCHAR, description VARCHAR)")
        for ds in datasets:
            con.execute("INSERT INTO datasets VALUES (?,?)",
                        [ds.name, getattr(ds, "description", "") or ""])
    if categories is not None:
        con.execute("CREATE TABLE categories (name VARCHAR, description VARCHAR)")
        for c in categories:
            con.execute("INSERT INTO categories VALUES (?,?)",
                        [c.name, getattr(c, "description", "") or ""])
    return con
```

- [ ] **Step 4: Run to verify pass**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest tests/test_agent/test_metadata_catalog.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/metadata_catalog.py tests/test_agent/test_metadata_catalog.py
git commit -m "feat: in-memory catalog build helpers for metadata Q&A"
```

---

## Task 3: `retrieve_metadata_catalog` strategy (SQL + citations + fallback)

**Files:**
- Modify: `src/agent/strategies/metadata_catalog.py`
- Test: `tests/test_agent/test_metadata_catalog.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent/test_metadata_catalog.py`:

```python
import pytest
from src.agent.strategies import metadata_catalog as mc


class _Store:
    def __init__(self, docs):
        self._docs = docs
    async def list_documents(self, user_groups=None):
        if user_groups is None:
            return self._docs
        return [d for d in self._docs if any(g in d.acl_groups for g in user_groups)]
    async def list_datasets(self, active_only=True):
        return [SimpleNamespace(id=2, name="DoD", description="")]
    async def list_categories(self):
        return [SimpleNamespace(name="payroll", description="")]


def _state(q="how many pdfs?", groups=("executives",)):
    return {"question": q, "user_groups": list(groups), "retrieval_attempts": 0}


@pytest.mark.asyncio
async def test_aggregate_returns_sql_rows(monkeypatch):
    store = _Store([_doc(doc_id="a"), _doc(doc_id="b")])
    def fake_sql(schema_prompt, question, generate_fn=None):
        return "SELECT COUNT(*) AS n FROM files"
    monkeypatch.setattr(mc, "generate_sql", fake_sql)
    out = await mc.retrieve_metadata_catalog(_state(), metadata_store=store)
    assert out["sql_results"] == [{"n": 2}]
    assert out["structured_trace"]["status"] == "ran"
    assert out["retrieved_chunks"] == []           # aggregate -> no file citations


@pytest.mark.asyncio
async def test_per_file_rows_yield_citation(monkeypatch):
    store = _Store([_doc(doc_id="a", filename="pay.pdf")])
    def fake_sql(schema_prompt, question, generate_fn=None):
        return "SELECT filename, doc_id FROM files"
    monkeypatch.setattr(mc, "generate_sql", fake_sql)
    out = await mc.retrieve_metadata_catalog(_state("list files"), metadata_store=store)
    assert len(out["retrieved_chunks"]) == 1
    assert out["retrieved_chunks"][0].metadata.doc_id == "a"
    assert out["retrieved_chunks"][0].metadata.filename == "pay.pdf"


@pytest.mark.asyncio
async def test_acl_scopes_documents(monkeypatch):
    visible = _doc(doc_id="a")          # acl ["executives"]
    hidden = _doc(doc_id="b")
    hidden.acl_groups = ["admins"]
    store = _Store([visible, hidden])
    def fake_sql(schema_prompt, question, generate_fn=None):
        return "SELECT COUNT(*) AS n FROM files"
    monkeypatch.setattr(mc, "generate_sql", fake_sql)
    out = await mc.retrieve_metadata_catalog(_state(groups=("executives",)), metadata_store=store)
    assert out["sql_results"] == [{"n": 1}]     # only the executives-visible doc


@pytest.mark.asyncio
async def test_sql_error_falls_back_to_snapshot(monkeypatch):
    store = _Store([_doc(doc_id="a"), _doc(doc_id="b")])
    def boom(schema_prompt, question, generate_fn=None):
        raise RuntimeError("bad sql")
    monkeypatch.setattr(mc, "generate_sql", boom)
    out = await mc.retrieve_metadata_catalog(_state(), metadata_store=store)
    assert out["sql_results"] == []
    assert out["structured_trace"]["fell_back"] is True
    assert len(out["retrieved_chunks"]) == 1
    snap = out["retrieved_chunks"][0]
    assert snap.metadata.doc_id == "metadata-context"
    assert "Total documents" in snap.text
```

- [ ] **Step 2: Run to verify they fail**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest tests/test_agent/test_metadata_catalog.py -k "aggregate or citation or acl or snapshot" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'retrieve_metadata_catalog'` (and `generate_sql` not imported).

- [ ] **Step 3: Implement the strategy**

Append to `src/agent/strategies/metadata_catalog.py`:

```python
from collections import Counter

from src.agent.strategies.structured import generate_sql, StructuredLookupTrace
from src.ingestion.tabular_store import execute_duckdb_sql
from src.retrieval.models import RetrievedChunk, ChunkMetadata

_ALLOWED = {"files", "datasets", "categories"}


def _citations_from_rows(rows, docs) -> list:
    """Build file citations for result rows that name a real document (doc_id)."""
    by_id = {d.doc_id: d for d in docs}
    chunks, seen = [], set()
    for r in rows:
        did = r.get("doc_id")
        if did and did in by_id and did not in seen:
            seen.add(did)
            d = by_id[did]
            chunks.append(RetrievedChunk(
                text=f"{d.filename} (type: {d.doc_type}): {getattr(d, 'summary', '') or ''}",
                score=0.9,
                metadata=ChunkMetadata(
                    doc_id=d.doc_id, filename=d.filename, doc_type=d.doc_type,
                    chunk_index=0, start_char=0,
                    acl_groups=getattr(d, "acl_groups", []) or []),
            ))
    return chunks


def _catalog_snapshot_chunk(docs) -> RetrievedChunk:
    """ACL-filtered prose snapshot of the catalog: totals + rollups + a file list."""
    by_type = Counter(d.doc_type for d in docs)
    lines = [f"Total documents you can access: {len(docs)}",
             "By type: " + ", ".join(f"{k}: {v}" for k, v in sorted(by_type.items()))]
    for d in docs[:50]:
        lines.append(f"- {d.filename} (type {d.doc_type}, {getattr(d, 'chunk_count', 0) or 0} chunks)")
    return RetrievedChunk(
        text="Document catalog (metadata):\n" + "\n".join(lines),
        score=0.3,
        metadata=ChunkMetadata(
            doc_id="metadata-context", filename="metadata_context",
            doc_type="metadata", chunk_index=0, start_char=0, acl_groups=["ALL"]),
    )


async def retrieve_metadata_catalog(state, metadata_store=None, generate_fn=None) -> dict:
    """Answer a catalog question via text-to-SQL over the ACL-filtered in-memory
    catalog. Falls back to a prose snapshot on SQL error or empty result."""
    question = state["question"]
    user_groups = state.get("user_groups", [])
    if metadata_store is None:
        from src.api.routes_ingest import get_metadata_store
        metadata_store = get_metadata_store()

    docs = await metadata_store.list_documents(user_groups)   # ACL boundary
    try:
        datasets = await metadata_store.list_datasets()
    except Exception:
        datasets = []
    try:
        categories = await metadata_store.list_categories()
    except Exception:
        categories = []
    dataset_names = {getattr(ds, "id", 0): ds.name for ds in datasets}

    trace = StructuredLookupTrace(query_type="metadata")
    rows = []
    try:
        con = build_catalog_connection(docs, dataset_names, datasets, categories)
        try:
            trace.sql = generate_sql(CATALOG_SCHEMA, question, generate_fn=generate_fn)
            rows = execute_duckdb_sql(con, trace.sql, allowed_tables=_ALLOWED)
            trace.status = "ran"
            trace.rows = rows
            trace.row_count = len(rows)
            trace.sample_rows = rows[:5]
        finally:
            con.close()
    except Exception as e:
        trace.status = "error"
        trace.error = str(e)
        logger.warning("Metadata catalog SQL failed: %s", e)

    attempts = state.get("retrieval_attempts", 0) + 1
    if trace.status == "ran" and rows:
        return {"retrieved_chunks": _citations_from_rows(rows, docs),
                "sql_results": rows, "structured_trace": trace.to_dict(),
                "retrieval_attempts": attempts}

    # Fallback: prose snapshot so we answer "what I can see" instead of "no data".
    trace.fell_back = True
    return {"retrieved_chunks": [_catalog_snapshot_chunk(docs)],
            "sql_results": [], "structured_trace": trace.to_dict(),
            "retrieval_attempts": attempts}
```

- [ ] **Step 4: Run to verify pass**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest tests/test_agent/test_metadata_catalog.py -v`
Expected: PASS (all helper + strategy tests).

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/metadata_catalog.py tests/test_agent/test_metadata_catalog.py
git commit -m "feat: retrieve_metadata_catalog strategy with citations and snapshot fallback"
```

---

## Task 4: Graph wiring + deploy/e2e

**Files:**
- Modify: `src/agent/graph.py`
- Test: end-to-end (manual)

- [ ] **Step 1: Add the routing branch**

In `src/agent/graph.py`, in the retrieve node's `elif` chain (after the `QueryType.ANALYTICAL` branch at line ~136), add:

```python
        elif query_type == QueryType.METADATA:
            from src.agent.strategies.metadata_catalog import retrieve_metadata_catalog
            result = await retrieve_metadata_catalog(retry_state, metadata_store=metadata_store)
```

(`metadata_store` is captured by the node closure from `create_agent_graph`; `retrieve_metadata_catalog` falls back to `get_metadata_store()` if it is None.)

- [ ] **Step 2: Run the agent test suite (no regressions)**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest tests/test_agent -q`
Expected: the new tests pass; pre-existing environmental failures (test_graph/test_lookup/test_sweep/test_cross_reference — need live backends) unchanged. No NEW failures.

- [ ] **Step 3: Commit**

```bash
git add src/agent/graph.py
git commit -m "feat: route METADATA queries to the catalog strategy"
```

- [ ] **Step 4: Deploy**

```bash
docker compose build api && docker compose up -d api
```
Expected: api healthy.

- [ ] **Step 5: End-to-end checks (deployed)**

```bash
docker exec -i sauron-api-1 env PYTHONPATH=/app python - <<'PY'
import json, urllib.request
from src.auth.jwt import create_token
tok = create_token("verify", ["executives"])
for q in ["How many PDFs do we have?",
          "When was 2025_April_Dec_AD_Pay.pdf uploaded?",
          "Which files mention officers?",
          "What datasets exist?"]:
    req = urllib.request.Request("http://localhost:8080/api/v1/query",
        data=json.dumps({"question": q}).encode(),
        headers={"Content-Type":"application/json","X-API-Key":"dev-key-1","Authorization":f"Bearer {tok}"},
        method="POST")
    print("Q:", q)
    print("A:", json.load(urllib.request.urlopen(req, timeout=300))["answer"][:200], "\n")
PY
docker logs sauron-api-1 2>&1 | grep -iE "Classified .*metadata|Text-to-SQL" | tail -8
```
Expected: each question logs `Classified ... -> metadata (tables_available=...)`, the answers contain exact counts / the upload date / matching filenames / dataset names. Report results.

---

## Notes for the implementer

- YAGNI: read-only catalog Q&A only — no new extraction, no admin/write ops, no joining the catalog to the DuckDB content tables.
- ACL is enforced solely by `list_documents(user_groups)`; never bypass it. The SQL `allowed_tables` set is `{files, datasets, categories}`.
- Reuse `generate_sql`/`execute_duckdb_sql`/`StructuredLookupTrace`/`RetrievedChunk` — do not reimplement SQL handling or trace types.
- `metadata-context` is already in `SYNTHETIC_IDS`, so the snapshot chunk is score-exempt in the synthesizer.
