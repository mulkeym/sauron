# Robust PDF data extraction — design

**Date:** 2026-05-29
**Status:** Approved (brainstorming) — ready for implementation plan
**Related:** `2026-05-25-tabular-spreadsheet-ingestion-design.md` (the Excel tabular pipeline this reuses), `2026-05-27-extensible-table-hints*` (the hint/glossary mechanism this reuses)

## Problem

PDFs with tabular data are ingested as low-quality flat text, so their data is effectively unqueryable and often unfindable.

Concrete case that motivated this: `2025 April Dec AD Pay.pdf` (U.S. military Active-Duty monthly pay chart) was ingested but never surfaced for the question *"What is the pay range for an enlisted person?"* Investigation (read-only, against the deployed app) found:

- The PDF **was** ingested (doc `dcfd5b40-6806-4c7a-be52-a386e2e2eb19`, 16 chunks across size tiers, ACL `["executives"]`, category `payroll_compensation`). Access was **not** the blocker — the same ACL group answers the GS/officer Excel questions.
- PDFs are **excluded** from the tabular pipeline (`SPREADSHEET_DOC_TYPES = ("xlsx","xls","csv","tsv")` in `tabular_ingest.py:193`), so the file got no DuckDB rows, no schema, no row narratives.
- `parser.py:_parse_pdf` uses `pypdf` → one flat text blob. The pay grid was extracted as a **wall of numbers with the structure destroyed**: rows split mid-table, rank labels sheared off (a chunk literally begins `30.90 $9,261.90 …`), numbers cut mid-value. No chunk keeps *rank + years-of-service + dollar amount* together.
- The word **"enlisted" appears only in the document summary**, never in the data rows (which say `E-1…E-9`). So a semantic query for "enlisted" has almost nothing to match in the chunks that hold the pay numbers, and there is no mapping from "enlisted" to the `E-*` paygrades.

So the miss had two root causes: **(a)** the table was not extracted into a usable structure, and **(b)** the data is not *findable* by the words users actually search. This project addresses both.

## Goals

- Robustly extract **all** data from PDFs — both **tables** and **prose** — into the structures that make them queryable and findable.
- Tables become first-class queryable data (DuckDB rows + schema + deterministic row narratives + correct citations), identical to the Excel path.
- Handle the common case (digital PDFs with an embedded text layer) well, and the rarer case (scanned/image PDFs) via OCR.
- Make extracted tabular data findable by user vocabulary via a code/class glossary (e.g. "enlisted" → `E-1…E-9`).
- Run **fully offline** after build (no model/binary downloads at runtime).
- Never regress ingestion: any failure falls back to today's flat-text behavior.

## Non-goals / explicitly deferred

- **Auto-derivation of glossaries** for arbitrary future code schemes (LLM proposes a code→meaning mapping, operator approves). This is the reserved Phase 2/3 of the hint roadmap; this project seeds **one curated, verified** glossary and builds the mechanism, but does not auto-generate glossaries.
- General document-layout reconstruction beyond prose + tables (figures, charts, form fields).
- A migration/re-ingest script for existing PDFs — the operator will wipe all data and re-ingest from scratch.
- Changing the retrieval/routing logic itself. This project changes *what gets stored*; existing classification/retrieval consume it unchanged.

## Constraints / decisions (from brainstorming)

- **PDF variety:** primarily digital PDFs (tables + prose); some scanned/image PDFs need OCR. Build for both, common case is digital.
- **Dependencies:** heavier image is acceptable, but the image must be **fully offline** — all models/binaries baked at build time, no runtime fetches.
- **Scope:** extraction **and** findability (the glossary) in this one spec.
- **Engine approach (chosen): Approach B — best-tool-per-job hybrid** behind a swappable `PdfExtractor` interface:
  - digital tables → `pdfplumber` (precise cell grids for ruled tables),
  - digital prose → `pdfplumber`/`pypdf` text,
  - scanned/image pages → `unstructured` hi_res + tesseract OCR.
  - Rationale: robustness on dense numeric tables is the priority, and that is where a single layout-model engine (`unstructured` hi_res) is weakest (cell mis-alignment on wide numeric grids); OCR is confined to the rare scanned case.

## Spike results (2026-05-29)

Validated Approach B against the real `2025 April Dec AD Pay.pdf` with pdfplumber 0.11.9:
- The chart is a **single page**; default `extract_tables()` (lattice/ruled) returns one **35×23** table.
- Rank labels (`O-10`, `O-8`, `E-*`) are intact in **column 0**; the `2 or less / Over 2 … Over 40` year-of-service headers are intact; dollar values are aligned and **whole** (no shearing). Engine choice confirmed — no revisit needed.
- **Default lattice is the right primary strategy** (text-strategy stays as fallback for borderless tables).
- The grade column header is **blank**, so `_safe_column_names` will name it `col_0` → the Task 16 glossary seed should target `--column col_0` (confirm against the registered schema post-ingest).
- Multi-page stitching is **not** needed for this file (single page) but is retained for multi-page tables in other PDFs.

## Architecture

### Common shape

`SheetGrid` (in `tabular.py`) is just `{sheet_name: str, rows: list[list]}`. The spreadsheet branch in `pipeline.py` already turns grids into DuckDB tables + schema + row narratives, and routes leftover text through the chunker. Therefore every PDF approach converges on: **produce prose blocks + table grids from a PDF, then feed the existing tabular pipeline + text chunker.**

### New module: `src/ingestion/pdf_extract.py`

Owns all PDF→content logic behind one interface:

```python
@dataclass
class ProseBlock:
    text: str
    page: int

@dataclass
class ExtractedPdf:
    prose_blocks: list[ProseBlock]   # paragraphs/headings, page-tagged
    table_grids: list[SheetGrid]     # one per detected (and stitched) table
    method: str                      # "digital" | "ocr" | "mixed"  (observability)

def extract_pdf(path: Path) -> ExtractedPdf:   # single entry point
    ...
```

**Per-page triage** (a PDF may mix digital and scanned pages):

1. **Digital check** — open with `pdfplumber`; a page with extractable characters above a small threshold is *digital*, otherwise *scanned*.
2. **Digital page** — `pdfplumber.extract_tables()` → grids; `pdfplumber.extract_text()` with table bounding boxes subtracted → prose (so prose is not duplicated inside a table).
3. **Scanned page** — `unstructured.partition_pdf(strategy="hi_res", infer_table_structure=True)` (OCR via tesseract); `Table` elements' `metadata.text_as_html` → grids; `NarrativeText`/`Title`/`ListItem` → prose.

`method` records `"digital"`, `"ocr"`, or `"mixed"` for the document.

### Table extraction → `SheetGrid` normalization

- **Strategy:** try `pdfplumber` `lattice` (lines-based, best for ruled tables) first; fall back to `stream` (whitespace clustering) if lattice yields nothing/garbled. (The spike confirms which the AD chart needs.)
- **Normalize:** each detected table → `SheetGrid(sheet_name="p<page>_table<n>", rows=...)`; strip fully-empty rows/cols; `None`→`""`. Then the **existing** `classify_sheet` decides clean vs messy — no new classification.
- **Multi-page stitching (`stitch_tables`):** consecutive tables on adjacent pages whose **header row matches** are merged into one grid, dropping the repeated header on continuation pages. (The AD chart repeats `2 or less … Over 40` across officer/warrant/enlisted pages.) Non-matching headers stay separate.
- **Cell-integrity guard:** if a data row's cell count diverges from the header's (a sheared-table signal), the table is **demoted to messy-region narratives** rather than loaded as a corrupt DuckDB table. Fail-safe, never silently wrong; logged.

### Prose extraction

- Digital: per-page text with table bounding boxes subtracted (`page.outside_bbox(...)` / filtered words) → `ProseBlock`s, page-tagged. De-duplicates prose vs table content.
- Scanned: prose elements from `unstructured`.
- All prose joins into `prose_text` and flows through the **normal `chunk_text`** multi-tier chunker (continuous prose, not the row-atomic `build_tier_chunks`).

### Pipeline integration (refactor)

Today `ingest_structured_sheets(file_path, …)` calls `read_sheets()` internally. Split grid *acquisition* from grid *ingestion*:

- `ingest_grids(grids, prose_text, doc_id, …)` — the existing clean→DuckDB / messy→narratives loop, **plus** routing `prose_text` through `chunk_text`.
- **Excel path:** `read_sheets(file_path)` → `ingest_grids(grids, prose_text="")`.
- **PDF path:** `extract_pdf(file_path)` → `ingest_grids(table_grids, prose_text=joined_prose)`.

Add a doc-type set so `pdf` enters this branch (e.g. a `STRUCTURED_DOC_TYPES` that includes spreadsheets + `pdf`, or a parallel `is_pdf` branch). This change must land in **both** `pipeline.py` (sync) and `queue.py` (async worker) — both ingest, and divergence has bitten before (see "two ingestion paths").

### Findability — paygrade/code glossary

Reuses the existing `HintStore` / `SchemaHint` / `resolve_hints` mechanism (the locality `TU`→Tampa glossary).

1. **Into row narratives (fixes vector retrieval):** thread the resolved column value-glossary into `build_row_narratives` so the key-column value is annotated — e.g. *"…pay grade **E-3 (Enlisted Member)**, over 4 years of service: $3,081.00."* Now "enlisted" is in the embedded text.
2. **Into the SQL schema prompt (fixes analytical queries):** ensure PDF-derived tables get hints resolved like Excel tables, so the existing `schema_prompt_with_values(..., hints=)` annotates values as `CODE (meaning)` and generated SQL can map "enlisted" → `WHERE grade LIKE 'E-%'`.
3. **Prefix-pattern support:** extend `value_glossary` to support patterns (`E-*`→Enlisted, `O-*`→Commissioned Officer, `W-*`→Warrant Officer) — far less brittle than enumerating every grade. Exact-match entries still work for non-systematic schemes.
4. **Seed (curated, not guessed):** seed one verified military-paygrade glossary scoped to `category=payroll_compensation` via the existing `/admin/api/hints/bulk`. (The locality lesson: the model's guess "TU=Tampa" was wrong — glossary content must be operator-verified.)

## Error handling

Fail-open at every layer, matching the existing pipeline:

- `extract_pdf` raises → fall back to today's `pypdf` flat text → text chunks.
- A single table fails the integrity guard → that table becomes messy-region narratives; the rest of the document proceeds.
- OCR error on a scanned page → that page degrades to whatever text is available (or is skipped), logged.
- A PDF never fails to ingest because of the new path; worst case it equals current quality. Every fallback logs **why** (no silent degradation).

## Offline build (hard constraint)

In the `Dockerfile`:

- `apt-get install tesseract-ocr poppler-utils`.
- Pre-download `unstructured` hi_res layout + table-transformer model weights at **build time**; set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` so nothing is fetched at runtime.
- A build-time smoke test extracts a tiny bundled PDF and **fails the build** if models are not resident — guarantees the image is genuinely offline-complete.
- New Python dep: `pdfplumber` (pure-Python).

## Re-ingestion

**Out of scope.** Existing PDFs only benefit after re-ingestion, but the operator will wipe all data and re-ingest from scratch — no migration/re-ingest script is built here. DuckDB single-writer still relies on `max_parallel_ingestion=1` (unchanged).

## Testing

- **Spike first (implementation task 1):** run `pdfplumber` on the real `2025 April Dec AD Pay.pdf`; confirm the ~20-column grid and multi-page stitch reconstruct correctly. Validates Approach B before building the rest. If pdfplumber cannot reconstruct this grid, revisit the engine choice before proceeding.
- **Unit (small fixture PDFs):** digital table → grid; prose/table de-duplication; multi-page `stitch_tables`; cell-integrity fallback to messy; per-page digital-vs-scanned triage.
- **OCR path:** one tiny scanned fixture (marked slow).
- **Glossary:** prefix-pattern resolution; narrative annotation ("E-3 (Enlisted Member)").
- **Integration:** PDF → `ingest_grids` → DuckDB rows + row narratives + prose chunks; assert an "enlisted" query retrieves the right rows and the answer cites the PDF filename.
- All via TDD. Both `pipeline.py` and `queue.py` paths covered.

## Success criteria

- `2025 April Dec AD Pay.pdf`, re-ingested, produces DuckDB rows + schema + row narratives for the pay grid (not just flat-text chunks).
- "What is the pay range for an enlisted person?" retrieves the `E-*` rows and answers with a range, citing the PDF by filename.
- Digital PDFs with prose + tables, and scanned PDFs, both ingest without error; failures degrade to flat text with a logged reason.
- The image runs the full extraction path with no network access.

## Open questions for the plan

- Exact `value_glossary` pattern syntax (`E-*` glob vs regex) and where pattern-matching slots into `resolve_hints`.
- Whether to add a single `STRUCTURED_DOC_TYPES` set or a parallel PDF branch in the two ingestion entry points.
- Stitch key precision: header-equality only, or also column-count/position, to avoid merging unrelated adjacent tables.
