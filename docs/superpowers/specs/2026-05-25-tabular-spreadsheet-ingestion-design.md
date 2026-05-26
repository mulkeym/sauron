# Tabular Spreadsheet Ingestion & Retrieval

## Overview

Make spreadsheets (.xls/.xlsx/.csv/.tsv) first-class, queryable data instead of opaque text blobs. Each *sheet* is classified at ingestion as a **clean data table** or **messy/narrative**, and routed accordingly: clean tables are loaded into a queryable DuckDB store with an auto-registered schema and answered by exact text-to-SQL; messy sheets get structure-aware chunking and the existing (Phase-0-hardened) RAG path. A deterministic row-narrative layer gives tabular rows real semantic signal for fuzzy retrieval, while the raw row stays the source of truth for the answering model.

This is scoped for **balanced speed + correctness** across a **mixed** future corpus: we invest work at ingestion so queries are both fast and exact, and we never assume a file's shape in advance.

## Problem

### What happened (the triggering incident)

A query — *"What are the pay scales for an engineer? What would my pay range be in Tampa, FL?"* — ran for **~2600s (43 min)** and returned an answer with 15 documents dropped as "could not be analyzed." Confirmed from container logs (the 20:30 query):

```
20:30:39  MAP first pass starts (27 docs @ concurrency 8)
20:47:55  23 docs failed → retry starts            ► FIRST PASS  = 1036s
20:47:55  RETRY pass (23 docs @ concurrency 4)
21:13:49  0/27 usable; 15 still failed             ► RETRY PASS  = 1554s
          MAP + retry total = 2590s ≈ 2600s
```

Root causes confirmed empirically:

1. **Spreadsheets are flattened to pipe-delimited text** (`header | val | val`), chunked into size tiers, and embedded. No structure is preserved. `get_chunks_by_doc(doc_id, 200, "large")` then rejoins them; with the live setting `llm_max_context=300000`, the GS pay tables (measured **170K–267K chars** each) are sent **untruncated** — ~67K tokens of number grid — to a local quantized 26B Gemma (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`), asking for up to 8192 output tokens. Each call runs to the **300s timeout**.
2. **The retry pass re-ran deterministic (size-driven) failures at half concurrency** (`retry_concurrency = llm_concurrency // 2 = 4`), producing **0** usable results and wasting ~1554s.
3. **The structured/SQL path already exists but is dormant.** `retrieve_analytical` does text-to-SQL, but `schema_registry` is *always empty* (ingestion never registers anything), so analytical queries silently fall back to map-reduce. The right tool for "GS-12 step 5 in Tampa" is built but unwired.

### Why this recurs

Embedding raw number rows is near-useless (`GS-12 | 5 | 86415` carries no signal a natural-language query can hit), and feeding whole number grids to an LLM is both slow and error-prone. As more spreadsheets arrive, the map-reduce path gets slower and less accurate. We need tabular data treated as data.

## Runtime context (verified)

| Setting | Value in the running container |
|---|---|
| LLM model | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (local, vLLM @ `host.docker.internal:8000`) |
| `vllm_request_timeout` | 300s |
| `llm_concurrency` | 8 (retry currently runs at 4) |
| `llm_max_context` | 300000 chars |
| Vector store | LanceDB, table `chunks` (`lancedb_path`, `lancedb_table_name`) |
| Analytical DB | async SQLAlchemy against `DATABASE_URL` (`sqlite+aiosqlite:///./data/metadata.db`) |
| Schema registry | `src/db/schema_registry.py`, in-memory, **always empty** |

The local model is slow and quantized: per-row LLM work does **not** scale, and any single oversized LLM call risks the 300s ceiling. The design avoids both.

## Solution

Two routing points — ingestion-time storage routing, and query-time question routing.

```
.xls/.xlsx/.csv  →  structured parse (per sheet)  →  clean vs. messy?
                                                       │
              ┌────────────────────────────────────────┘
   CLEAN TABLE │                                    MESSY/NARRATIVE
              ▼                                            ▼
  • rows → DuckDB table (queryable)             • structure-aware chunks
  • 1 LLM call/table → column labels,             (header repeated per chunk,
    key/measure cols, table description           rows never split mid-row,
  • register + PERSIST schema                     sheet boundaries marked)
  • deterministic row narratives → embed        • optional restate-only
    (raw row carried as metadata)                 row-GROUP narratives
  • embed table description (routing)           • structured profile in metadata
              │                                            │
              ▼                                            ▼
       ANALYTICAL / text-to-SQL  ◄─ query router ─►  (Phase-0-fixed) sweep +
       (DuckDB, exact)                                 map-reduce (RAG)
```

### Phase 0 — Map-reduce safety fixes (independent; ship first)

These bound latency for *any* large document and are the safety net under the messy-table path. Independent of the spreadsheet work.

In `src/agent/strategies/map_reduce.py`:

1. **Classify failures.** `map_document` tags a failure as `transient` (e.g. connection error) or `permanent` (timeout / size-driven). The retry loop in `_map_documents` retries **only** `transient` failures. **Timeouts are never retried** — they are deterministic and were the entire 1554s waste.
2. **Retry at full concurrency** (not half), if a retry happens at all.
3. **Per-MAP payload cap.** New setting `map_doc_char_budget` (≈ 60–80K chars), much tighter than `llm_max_context=300000`. `map_document` truncates content to this budget so no single MAP call can run to the 300s ceiling. Truncation is logged and surfaced in the incomplete-note mechanism that already exists.

Phase 0 does **not** make spreadsheets answerable well — that's the rest of the design. It only stops the 40-minute timeouts and the wasted retries.

### Component 1 — Structured parse + clean/messy classification

New module: `src/ingestion/tabular.py`.

- **Structured parse.** For tabular types, stop flattening straight to pipe-text in `parser.py`. Extract each sheet as a structured table (header + typed rows) via the existing libs (`openpyxl` for OOXML, `xlrd` for legacy `.xls`, `csv` for delimited — reuse the magic-byte detection already in `parser.py`). Includes **header-row detection** (header is not always row 1 in messy files).
- **Clean/messy classifier (heuristic, no LLM).** A sheet is `CLEAN_TABLE` when all hold: a detectable header row (mostly unique non-numeric labels), consistent per-column dtype down the rows, rectangular shape (consistent column count, low empty/merged-cell ratio), and ≥ `min_table_rows` rows. Otherwise `MESSY`. Heuristics decide the *route*; no LLM call is needed to route.
- Output per sheet: a `SheetClassification` (route + detected header + column dtypes), consumed by the pipeline branch.

### Component 2 — CLEAN tables → DuckDB store + schema + narratives

New modules: `src/ingestion/table_profiler.py`, `src/ingestion/tabular_store.py`.

- **One LLM call per table** (`table_profiler`): given the header + a small sample of rows, produce (a) human-readable column descriptions, (b) a `key` vs `measure` tag per column, and (c) a one-paragraph table description. This is **O(tables), not O(rows)** — the key scale property. If the call fails/times out, the sheet **falls back to the messy path** (fail safe; never drop the file).
- **Load rows into DuckDB** (`tabular_store`): one table per sheet, deterministic table naming (`doc_<docid>_<sheet>`), ACL columns/tags carried for filtering. DuckDB is synchronous → run via `asyncio.to_thread` (the pattern already used for `generate`/`embed_query`).
- **Register + PERSIST the schema.** `schema_registry` is in-memory today; add a metadata-DB table (`registered_schemas`) storing each `TableSchema` (database, table, columns, description, `acl_groups`). Load all rows into the registry on startup; write on ingest; delete on document deletion.
- **Deterministic templated row narratives.** Built from `key` + `measure` columns with no LLM and no derived math — e.g. *"In locality {locality}, a GS-{grade} step {step} employee's annual base pay is ${value} ({year} General Schedule)."* Zero LLM cost, zero hallucination, reproducible, scales to 100K+ rows. Granularity is configurable (`narrative_granularity`: `row` or `row_group`, default `row`). Each narrative is embedded into LanceDB with:
  - the **raw row** carried as metadata (the source of truth), and
  - a pointer to the DuckDB table + row identity, and
  - `doc_type="table_row"`.
- **Embed the table description** as its own chunk (`doc_type="table_summary"`) for table discovery/routing.

Templating rules: missing/null key or measure cells are rendered explicitly (e.g. "(not specified)") — never fabricated, never silently dropped.

### Component 3 — MESSY/narrative tables → improved RAG

- **Structure-aware chunking** (in `chunker.py` or a tabular-aware helper): repeat the detected header atop each chunk, **never split mid-row**, mark sheet boundaries, keep chunks modest in size.
- **Optional restate-only row-GROUP narratives** for the most table-like regions of a messy sheet (bounded LLM cost: these files are smaller/fewer than clean tables; cost is O(row-groups)). Constrained to restate cell values only — no trends, no aggregation.
- Store a structured profile (sheet names, detected columns, value ranges) in `metadata_tags`.
- Answered via the existing sweep + (Phase-0-fixed) map-reduce path.

### Component 4 — Query-time routing

- `src/agent/classifier.py`: **feed the list of registered table descriptions** (table name + one-line description, ACL-filtered for the user) into the classifier prompt, so it knows structured tables exist and can pick `ANALYTICAL` for questions like "GS-12 step 5 in Tampa." Today the prompt says "analytical … only if database tables exist" and none ever did.
- `ANALYTICAL` → `retrieve_analytical` → text-to-SQL. Add `execute_duckdb_sql` (sync, wrapped in `to_thread`) alongside the existing SQLAlchemy `execute_sql`; route table-backed queries to DuckDB. Keep the existing **SELECT-only validation**.
- **Fuzzy/semantic** tabular questions: normal vector search retrieves `table_row` narrative embeddings; the **raw rows** (from metadata) are handed to the synthesizer. Narrative is for *finding*; raw row is the *truth*.
- **Fallbacks preserved**: bad/failed SQL → validation catch → fall back to RAG (already the behavior in `analytical.py`). Nothing silently drops.

## Data flow (clean table, end to end)

```
ingest GS .xls
  → parser: magic-byte detect → structured per-sheet tables
  → tabular.classify: CLEAN_TABLE (header + typed cols + rectangular)
  → table_profiler (1 LLM call): label cols, key={grade,step,locality,year},
      measure={base_pay}, table description
  → tabular_store: CREATE DuckDB table doc_<id>_<sheet>, insert rows (ACL-tagged)
  → schema_registry.register(...) + persist to registered_schemas
  → for each row: deterministic narrative → embed → LanceDB
      (doc_type=table_row, metadata={raw_row, duckdb_table, row_id})
  → embed table description (doc_type=table_summary)

query "GS-12 step 5 base pay in Tampa"
  → classify (sees registered tables) → ANALYTICAL
  → retrieve_analytical → text-to-SQL → execute_duckdb_sql
      SELECT base_pay FROM doc_<id>_<sheet>
      WHERE grade='GS-12' AND step=5 AND locality LIKE '%Tampa%'
  → exact row(s) → synthesize → answer (bounded latency)
```

## Error handling & failure modes

| Failure | Handling |
|---|---|
| Clean-table false positive (messy sheet mis-routed to SQL) | text-to-SQL produces garbage → SELECT-only validation / execution error → fall back to RAG |
| Table-profiling LLM call fails or times out | Sheet falls back to the messy/RAG path; file still ingested |
| Per-table LLM timeout | Never blocks the rest of the file's ingestion (per-sheet isolation) |
| Null / missing cells in narrative | Rendered explicitly ("(not specified)"); never fabricated |
| Schema registry empty for a user's ACL | `retrieve_analytical` falls back to RAG (existing behavior) |
| Oversized doc still reaches map-reduce | Phase-0 payload cap truncates; timeout not retried |

## Testing

- **Unit:** header-row detection; clean/messy classifier on fixtures (a GS `.xls`, a multi-sheet messy file, a CSV, a narrative-style sheet); deterministic narrative golden-strings including null/missing-cell handling (assert no fabricated values); `schema_registry` persistence round-trip (register → restart-load → query).
- **Phase 0:** `_map_documents` does not retry a `permanent`/timeout failure; retries a `transient` one at full concurrency; `map_document` enforces `map_doc_char_budget`.
- **Integration:** ingest a GS `.xls` → DuckDB table exists + schema registered + `table_row` narratives embedded; ask "GS-12 step 5 base pay" → classified `ANALYTICAL` → exact SQL answer; assert **bounded latency** (well under the old 2600s; target < 60s).
- **Regression:** existing `tests/test_ingestion/test_parser_spreadsheet.py` and related parser/chunker/pipeline tests still pass.

## Scope / YAGNI (explicitly deferred)

- **On-the-fly code execution** (LLM writes pandas/SQL per query in a sandbox) — deferred.
- **Multi-database federation** and an **admin UI** for schemas — deferred.
- **LLM-authored per-*row* narratives** — rejected (does not scale on the local model; deterministic templating captures the benefit safely).
- **Reuse LanceDB/Lance as the structured row store** (DuckDB-over-Arrow, zero-copy) — attractive future optimization that would collapse to a single storage system, but it needs the Lance↔DuckDB pushdown + ACL path validated first. v1 uses dedicated DuckDB tables for the rows and LanceDB for the narrative embeddings.

## Open questions / decisions made

- **Tabular store engine:** DuckDB as the SQL executor (LanceDB is not a SQL engine — it does predicate filtering, not `GROUP BY`/joins/aggregation, which the mixed corpus needs). v1 stores rows in dedicated DuckDB tables; LanceDB continues to hold the narrative embeddings.
- **Narrative author:** deterministic templating (not LLM) for rows; one LLM call per *table* for column understanding + description; LLM row-*group* narratives only for messy sheets, restate-only.
