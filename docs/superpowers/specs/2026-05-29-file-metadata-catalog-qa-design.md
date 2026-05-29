# File-metadata (catalog) Q&A

**Date:** 2026-05-29
**Status:** Design — approved for planning

## Problem

The system answers questions from document **content** (lookup / sweep / analytical /
cross_reference / temporal), but cannot answer questions *about the documents
themselves* — the catalog. Questions like "how many PDFs do we have?", "when was
`2025_April_Dec_AD_Pay.pdf` uploaded?", "which files mention officers?", or "what
datasets exist?" currently route to content retrieval and fail (no content chunk
answers them). The existing `metadata-context` path only injects extracted tags as a
side-enrichment within SWEEP; it is not a way to query the catalog.

## Goal

Answer four classes of catalog question, ACL-scoped to the asking user:
1. **Catalog aggregates** — counts, lists, group-bys, filters ("how many PDFs?",
   "files uploaded in May", "docs per dataset", "most recent upload").
2. **Per-file facts** — facts about a named file ("when was X uploaded / what
   dataset / who uploaded it / how many chunks?").
3. **Content-tag search** — find files by extracted `metadata_tags` ("which files
   mention Acme?", "files about pay policy").
4. **Catalog overview** — what exists ("what datasets/categories/file types are
   there?").

All four are queries over the catalog (`documents`, and for overview the
`datasets`/`categories` metadata), not document content.

## Architecture

Recognition is automatic: a new `METADATA` query type the classifier routes to.
Answering is **text-to-SQL over an ACL-pre-filtered, in-memory DuckDB catalog** — the
SQL physically cannot see rows outside the user's ACL, which sidesteps the
`include_tables ≠ ACL` pitfall recorded in the prior LangChain-SQL evaluation.

### 1. Query type + routing

- Add `METADATA = "metadata"` to `QueryType` (`src/agent/state.py`).
- Extend `CLASSIFICATION_PROMPT` (`src/agent/classifier.py`) with the METADATA type
  and explicit contrast examples separating *about the file* from *in the file*:
  - METADATA: "how many documents are there?", "when was <file> uploaded?", "which
    files mention officers?", "what datasets exist?", "list PDFs uploaded in May".
  - NOT METADATA (content): "what does <file> say about officers?", "what is the pay
    for an O-4?".
- Strategy-Memory guard: extend the protection shipped in
  `2026-05-29-strategy-memory-protect-analytical` so a learned prior cannot override
  a METADATA pick either (METADATA is a deterministic, capability-gated pick like
  ANALYTICAL). The guard condition becomes `llm_pick in (ANALYTICAL, METADATA)`.

### 2. Strategy — `retrieve_metadata_catalog(state, metadata_store)`

New file `src/agent/strategies/metadata_catalog.py`.

1. `docs = await metadata_store.list_documents(user_groups)` → ACL-filtered records.
2. Build an ephemeral in-memory DuckDB connection with a `files` table, one row per
   visible document, columns:
   `doc_id, filename, doc_type, dataset, category, uploaded_by, created_at,
   chunk_count, summary, tags`.
   - `dataset` is the dataset **name** (resolved from `dataset_id` via
     `list_datasets`), not the numeric id, so SQL/answers are human-readable.
   - `tags` is a single lowercased text column built by flattening the existing
     `metadata_tags` dict (entities, organizations, amounts, topics, identifiers)
     into one searchable string — enables `tags ILIKE '%officer%'`. No new
     extraction; reuses what ingestion already stored.
   - For catalog-overview questions, also create small ACL-scoped `datasets` and
     `categories` tables (name + description), filtered to those the user can see.
3. Generate + run text-to-SQL against the catalog, reusing `generate_sql` / `run_sql`
   / `execute_duckdb_sql` with a catalog-specific schema prompt. Read-only `SELECT`,
   restricted to `{files, datasets, categories}`.
4. Return:
   - `sql_results`: the rows (the existing synthesizer already renders SQL rows).
   - `structured_trace`: a `StructuredLookupTrace(query_type="metadata", ...)` for the
     playground (SQL, row_count, status).
   - `citations`/`retrieved_chunks`: for per-file and content-tag answers, attach
     citations to the matched documents (by `doc_id`/`filename`) so the answer links
     to real files. Pure aggregates need no citation.

### 3. Graceful fallback

On SQL error, or an empty result where prose still helps (broad overview), fall back
to an ACL-filtered **catalog snapshot** chunk — the visible document list plus
precomputed rollups (counts by `doc_type`, `dataset`, `category`) — handed to the
synthesizer. Mirrors the analytical zero-row → structured fallback pattern. The
feature degrades to "here is what I can see," never "no data."

### 4. Graph wiring

Add a `QueryType.METADATA` branch in the retrieve node of `create_agent_graph`
(`src/agent/graph.py`) that calls `retrieve_metadata_catalog(state, metadata_store)`.
The graph already holds `metadata_store`.

## Components & boundaries

- `src/agent/strategies/metadata_catalog.py` — the new strategy: catalog build +
  text-to-SQL + fallback. Single responsibility; depends on `MetadataStore`,
  DuckDB, and the existing text-to-SQL helpers.
- `src/agent/state.py` — enum addition only.
- `src/agent/classifier.py` — prompt addition + one-token change to the guard
  condition.
- `src/agent/graph.py` — one routing branch.

## Testing

Unit (`tests/test_agent/`):
- Catalog build: a list of doc records → a `files` table with correct columns and a
  flattened `tags` string (including `metadata_tags` values).
- Aggregates: text-to-SQL returns exact `COUNT`/`GROUP BY` over the catalog.
- Per-file: a filename question yields the right row and a citation to that doc.
- Content-tag: `tags ILIKE` finds the expected files and cites them.
- ACL: a user sees only documents whose `acl_groups` overlap their groups
  (enforced because `list_documents(user_groups)` pre-filters).
- Fallback: forced SQL error → snapshot chunk returned, never empty.
- Classifier: representative "about the files" questions route METADATA; "in the
  files" questions route content types.
- Guard: a learned LOOKUP record does not override a METADATA pick (reason
  "protected").

End-to-end (deployed stack): "how many PDFs do we have?", "when was
`2025_April_Dec_AD_Pay.pdf` uploaded?", "which files mention officers?", "what
datasets exist?" — exact answers, ACL-scoped, with file citations where applicable.

## Decisions taken (defaults)

- **Guard extended to METADATA** (a) — confirmed default; deterministic picks are
  authoritative.
- **`tags` from existing `metadata_tags`** (b) — no new extraction.

## Out of scope

- New metadata extraction or new tag fields.
- Write/admin operations (this is read-only Q&A).
- Cross-corpus analytics beyond the catalog (e.g. joining file metadata to the
  tabular content tables).
