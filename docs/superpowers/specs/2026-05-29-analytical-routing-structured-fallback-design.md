# ANALYTICAL routing + resilient structured fallback

**Date:** 2026-05-29
**Status:** Design — approved for planning

## Problem

Questions that should be answered from a structured table (e.g. "What is the pay
range for an officer?" against the ingested Active-Duty pay PDF) are answered with
"the document does not contain the specific amounts" even though the data is
ingested, glossary-annotated, and present in DuckDB and the knowledge graph.

Two compounding defects, both confirmed in current code:

1. **Routing.** `format_available_tables` (`src/agent/classifier.py:30`) emits only
   `- <table>: <generic auto-profiled description>` (e.g. "financial values indexed
   by col_0"). The classifier cannot tell the table holds military pay, so it picks
   LOOKUP. LOOKUP (`src/agent/strategies/lookup.py`) searches only `tier="medium"`
   prose chunks and structurally cannot see the `tier="table_row"` pay narratives.
   The seeded `table_note`/`column_note` hints feed the SQL prompt and the
   synthesizer — never the classifier — so they don't influence routing.

2. **Fallback.** Even when a question routes ANALYTICAL, `retrieve_analytical`
   (`src/agent/strategies/analytical.py:29`) falls back to map-reduce **only** on a
   hard SQL error (`trace.status == "error"`). A query that runs but returns **0
   rows** (e.g. the quantized model emits `WHERE col_0='officer'` when the values
   are `O-1`…`O-10`) returns `sql_results: []` and nothing reaches the synthesizer.

The knowledge graph does run during LOOKUP (parallel `enrich` node) and carries
fuzzy officer/dollar prose, but it holds approximate "ranges from $X to $Y" entity
blurbs, not exact per-grade/per-years cells — the wrong tool for precise tabular
answers. The structured (DuckDB / ANALYTICAL) path is the right one.

## Goal

"What is the pay range for an officer?" routes ANALYTICAL, runs SQL over the pay
table, and returns a real `$X–$Y` answer citing the PDF. When the model's SQL
misses, the verified glossary-annotated row narratives ("Commissioned Officer" +
dollar values) carry the answer instead of "no specifics."

Non-goals: changing ingestion/annotation (already correct), graphing numeric
tables, or reworking LOOKUP's tier scope.

## Part A — Routing enrichment

Fold the resolved, ACL-filtered hint notes into the classifier's table view so the
LLM can route value/range questions to ANALYTICAL.

- In the async `classify_node` (`src/agent/classifier.py`), resolve hints for the
  user's schemas via `resolve_hints_for_schemas(schemas, hint_store, metadata_store)`
  (already async-available; same call `retrieve_analytical` uses) and pass them into
  `format_available_tables`.
- `format_available_tables` appends a **compact** note to each table line when a
  note exists, e.g.
  `- doc_…_table0: financial values indexed by col_0 — U.S. military active-duty basic pay; Commissioned Officer (O-*), Enlisted (E-*)`.

Guards:
- **ACL:** notes derive from the same `list_for_user` schemas — no cross-tenant leak.
- **Prompt size:** cap each table's appended note (~200 chars); only append when a
  note exists, so the no-hints case is byte-identical to today.
- **Over-routing:** keep the existing instruction ("only if the question asks for
  specific values, totals, or filtered rows that these tables contain") — we enrich
  the description, not loosen the rule.
- **Determinism:** preserve the existing sort-by-table-name so the prompt is
  run-to-run identical.

Scope: enrich the classifier view. Because both ANALYTICAL and SWEEP read the same
classifier output, this naturally improves SWEEP routing too; no SWEEP-specific
change.

## Part B — Resilient fallback

Make `retrieve_analytical` treat a zero-row SQL result as a miss and fall back to
the structured row-narrative path.

- After `run_structured_lookup`, branch on outcome:
  - `status == "error"` → existing map-reduce fallback (unchanged).
  - `status == "ran"` and `row_count == 0` → fall back to `retrieve_structured`
    (`src/agent/strategies/structured.py:240`): SQL + top-k `tier="table_row"`
    annotated narratives + relevance gate.
  - `status == "ran"` and `row_count > 0` → return rows as today.
- If `retrieve_structured` itself gates out (table not relevant / still empty),
  fall through to map-reduce so the query is never left with nothing.
- Preserve `structured_trace` on every branch so the playground "Structured Lookup"
  step still reflects what happened (record the fallback, e.g. `fell_back=True`).

Resulting chain: **clean SQL min/max → annotated table-row narratives → map-reduce.**

## Testing

Unit:
- `format_available_tables` includes the note when a hint exists and is unchanged
  when none exists; note is length-capped; output stays sorted/deterministic.
- `classify_query` returns ANALYTICAL for the officer question given an enriched
  table line, and does not over-route a generic prose question.
- `retrieve_analytical` falls back to `retrieve_structured` on `row_count == 0`,
  falls back to map-reduce when structured gates out, and returns SQL rows
  unchanged when `row_count > 0`. `structured_trace` present in every branch.

End-to-end:
- In the deployed container, ask "What is the pay range for an officer?" and confirm
  ANALYTICAL routing + a real `$X–$Y` answer citing `2025_April_Dec_AD_Pay.pdf`.

## Risks

- Quantized-model SQL remains flaky; Part B's narrative fallback is the mitigation,
  but a precise true min–max depends on the model producing correct SQL.
- Classifier prompt growth with many registered tables — bounded by the per-note cap
  and the existing per-user table filtering.
