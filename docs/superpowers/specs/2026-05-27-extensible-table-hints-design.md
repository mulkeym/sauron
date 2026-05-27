# Extensible Table Hints — Design (Phase 1: curated foundation)

**Status:** approved 2026-05-27.

## Problem

Text-to-SQL over ingested spreadsheets fails on columns that hold opaque codes.
The OPM General Schedule tables store locality as a cryptic `locname` code
(`TU`, `RUS`, `ATL`, `GS`, …) with **no human-readable name anywhere in the
ingested data** (one table per doc, no legend sheet; the full-text parse also
contains only codes). So a question like "GS salary rates in Tampa" forces the
model to *guess* that Tampa → `TU`; it variously emits `'Tampa'` (0 rows),
`'GS'` (wrong rows), or `'TU'` (correct). A stable prompt (the
`distinct_values` `ORDER BY` fix) makes the guess deterministic but not correct.

We will be ingesting many documents from many sources (OPM today; potentially
other DoD organizations and private enterprises later), so the remedy must be a
general, **extensible** mechanism for attaching domain knowledge ("table hints")
to tables — not an OPM-specific hardcode.

## Goals

- Inject curated domain knowledge into the text-to-SQL schema prompt so the model
  filters on the right codes/columns.
- **Curate once, reuse across many docs** from the same source (the OPM locality
  glossary applies to every GS/LEO table).
- A uniform data model that accommodates three authoring layers — curated, auto,
  learned — without future schema changes (hybrid). **This spec ships the curated
  layer only**; auto and learned are later plans.
- Fully additive and fail-open: with no hints in scope, the prompt and all
  existing behavior are byte-for-byte unchanged.

## Non-goals (this phase)

- Auto-derived hints from the profiler (Phase 2) and feedback-learned hints
  (Phase 3). The data model reserves `provenance` for them; no code here.
- Join hints / cross-table relationship hints. The `hint_type` enum can grow
  later; not needed for the locality problem.
- Sourcing the actual OPM locality code→name list. That content is supplied via
  the bulk-import authoring path; this spec is the mechanism, not the data.

## Existing context (what we build on)

- `src/db/schema_registry.py` — `SchemaRegistry` (in-memory, `register` /
  `list_for_user`) holding `TableSchema` (`table`, `description`, `columns`,
  `acl_groups`). Loaded at startup via `populate_schema_registry` in
  `src/ingestion/tabular_ingest.py`, wired into `create_agent_graph`.
- `src/db/models.py` — `DocumentRecord` already carries `category` (string;
  there is a curated `categories` table with description/acl_groups/routing
  keywords) and `dataset_id` (int, 0 = unassigned). These are the existing
  "collection" concepts hints bind to. No new taxonomy is invented.
- `src/ingestion/tabular_store.py::schema_prompt_with_values(schemas, con,
  max_distinct=100)` — renders each table for the text-to-SQL prompt:
  ```
  Table: <table>
  Description: <desc>
  Columns:
    - <col> (<dtype>): <description> | values: a, b, c
  ```
  This module is intentionally I/O-free (only the passed DuckDB connection); hint
  resolution (which reads the hint store and document records) stays **out** of it.
- `src/agent/strategies/structured.py::generate_sql` — builds the prompt via
  `schema_prompt_with_values` and calls the LLM (`temperature=0.0`,
  `seed=settings.llm_seed`). This is where resolved hints are passed in.

## Architecture

Approach A (source-scoped hint records + query-time prompt enrichment). Three
isolated units plus storage:

### 1. Data model — `SchemaHint`

Persisted in metadata.db, table `schema_hints`:

| field | type | meaning |
|---|---|---|
| `id` | int PK | surrogate key |
| `scope_type` | str | `"category"` or `"dataset"` — which collection concept it binds to |
| `scope_value` | str | category name, or `dataset_id` rendered as string |
| `hint_type` | str | `"value_glossary"` \| `"column_note"` \| `"table_note"` |
| `target_column` | str \| null | column name the hint applies to (null for `table_note`); matched by name across every table in scope |
| `payload` | JSON | `value_glossary`: `{"TU": "Tampa-St. Petersburg, FL", "RUS": "Rest of U.S."}`. notes: `{"text": "..."}` |
| `provenance` | str | `"curated"` \| `"auto"` \| `"learned"` (Phase 1 writes only `"curated"`) |
| `confidence` | float | 1.0 for curated; reserved lower values for auto/learned |
| `created_at` | datetime | audit |
| `created_by` | str | audit |

Dataclass `SchemaHint` mirrors `TableSchema`'s style in `src/db/schema_registry.py`.

**Precedence** when multiple hints share a `(hint_type, target_column)` within a
table's resolved scope set: `curated > learned > auto`; ties broken by higher
`confidence`, then most recent `created_at`. `table_note` (null target) hints are
not deduped against each other — all in-scope table notes are kept.

### 2. Storage & loading — `HintStore` + `MetadataStore` methods

- `MetadataStore` (`src/db/metadata.py`): startup DDL creates `schema_hints`
  (same migration path as other tables). New async methods: `save_hint(hint)`,
  `load_all_hints() -> list[SchemaHint]`, `delete_hint(id)`,
  `list_hints_for_scope(scope_type, scope_value) -> list[SchemaHint]`.
- `HintStore` (new, `src/db/hint_store.py`, parallels `SchemaRegistry`):
  in-memory; `register(hint)`, `for_scope(scope_type, scope_value) ->
  list[SchemaHint]`, `clear()`. Pure, no I/O.
- `populate_hint_store(metadata_store, hint_store) -> int` (in
  `src/ingestion/tabular_ingest.py`, beside `populate_schema_registry`): loads
  all persisted hints; logs "Loaded N hint(s)". Called in `main.py` lifespan
  immediately after `populate_schema_registry`.
- `HintStore` is constructed beside `SchemaRegistry` and threaded into
  `create_agent_graph(...)` so retrieval strategies can reach it.

### 3. Resolver — `src/agent/strategies/hint_resolver.py`

`resolve_hints(table_schema, doc_record, hint_store) -> ResolvedHints` — pure, no
I/O (caller supplies the `DocumentRecord`).

- Gather hints from both scopes the doc belongs to: `("category",
  doc_record.category)` and `("dataset", str(doc_record.dataset_id))`.
- Apply precedence/dedup per `(hint_type, target_column)`.
- Restrict `value_glossary` / `column_note` hints to columns that actually exist
  on `table_schema` (name match).
- Return a compact structure:
  ```python
  ResolvedHints(
      column_glossaries: dict[str, dict[str, str]],  # col -> {code: meaning}
      column_notes: dict[str, str],                   # col -> note text
      table_notes: list[str],
  )
  ```
- Empty/missing doc record, unknown scope, or malformed payload → that hint is
  skipped (never raises).

### 4. Injection — `schema_prompt_with_values(..., hints=None)`

Add an optional `hints: dict[str, ResolvedHints] | None` parameter keyed by table
name. The caller (`generate_sql` path) resolves hints per in-scope table and
passes the map in. Rendering, when a table has resolved hints:

- **value_glossary** annotates the existing `values:` line *in place*:
  `values: TU (Tampa-St. Petersburg, FL), RUS (Rest of U.S.), GS, …` — code and
  meaning together, exactly where the model picks the filter value. Codes with no
  glossary entry render bare (as today).
- **column_note** appends ` — <note>` to that column's description line.
- **table_note** adds a `Notes: <text>` line under `Description:` (one line per
  note).

When `hints is None` or a table has no resolved hints, output is **byte-identical
to today** (additive + backward-compatible; preserves the `distinct_values`
determinism guarantee).

### Query-time data flow

`retrieve_analytical` / `retrieve_structured` → for each in-scope `TableSchema`,
fetch its `DocumentRecord` (doc_id is recoverable from the table name via the
existing `doc_<doc_id>_<sheet>` scheme / schema metadata) → `resolve_hints(...)`
→ build `{table_name: ResolvedHints}` → pass into `schema_prompt_with_values` →
enriched prompt → `generate_sql`. All wrapped fail-open.

## Authoring (Phase 1)

- Thin admin API + UI in `src/admin/routes.py`, beside category management:
  create / edit / delete a `SchemaHint`.
- **Bulk JSON import** endpoint so a whole glossary loads at once — e.g. the OPM
  locality glossary as a single `value_glossary` hint scoped to
  `("category", "<OPM category name>")`. This is how the OPM code→name content
  enters the system; the content itself is supplied by an operator from an
  authoritative source.

## Phasing

- **Phase 1 (this spec):** data model, store + loading, resolver, injection,
  admin authoring + bulk import. Fully fixes the OPM-codes problem once the
  glossary is imported.
- **Phase 2 (later):** `table_profiler.py` emits `provenance="auto"` hints (e.g.
  flags short-code VARCHAR columns) at low confidence; curated overrides.
- **Phase 3 (later):** mine `feedback` / `strategy_memory` for value→meaning
  corrections, written as `provenance="learned"`.

## Error handling

Fail-open throughout, matching the structured-retrieval conventions:

- Unwired or empty `HintStore` → resolution yields nothing → prompt unchanged.
- Resolver wrapped: malformed payload, missing `DocumentRecord`, or unknown scope
  skips the hint, never breaks SQL generation.
- Injection never raises; an exception in the per-table resolve/inject step falls
  back to the un-enriched prompt for that table.

## Testing

- **Resolver** (pure): curated overrides auto on same `(hint_type,
  target_column)`; `category` + `dataset` scopes merge; column-name targeting
  drops hints for absent columns; missing doc record / unknown scope → empty;
  malformed payload skipped.
- **Injection** (pure): glossary annotates the `values:` line
  (`TU (Tampa-St. Petersburg, FL)`); column note appends to description; table
  note renders as `Notes:`; **`hints=None` → output byte-identical to baseline**
  (regression guard protecting existing behavior + the determinism fix).
- **Store round-trip**: `save_hint` / `load_all_hints` / `delete_hint`;
  `list_hints_for_scope`; bulk JSON import parses and persists.
- **End-to-end** (in-container, as validated this session): import the OPM
  locality glossary scoped to the OPM category → `generate_sql` for "GS salary
  rates in Tampa" deterministically emits `WHERE locname = 'TU'` and returns rows.

## Spec self-review

- **Placeholders:** none. (`<OPM category name>` is a runtime value supplied at
  import time, not an unresolved spec decision.)
- **Internal consistency:** data model fields used by resolver/injection match
  the table definition; `provenance` precedence stated once and reused; phasing
  matches goals/non-goals (curated only this phase).
- **Scope:** focused on the curated foundation — one implementation plan's worth.
  Auto/learned explicitly deferred.
- **Ambiguity:** scope binding is exactly two concepts (`category`, `dataset`);
  precedence and dedup keys are explicit; injection rendering is concrete per
  hint type; `hints=None` behavior is pinned to byte-identical.
