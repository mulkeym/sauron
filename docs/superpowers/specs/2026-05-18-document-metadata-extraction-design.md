# Universal Document Metadata Extraction for Optimized Retrieval

## Overview

Extract structured metadata from every document at ingestion time and use it to speed up sweep/map-reduce queries. Instead of the LLM reading every discovered document in full, the sweep strategy first searches metadata to narrow the document list, then only MAP-reads the truly relevant ones.

## Problem

The current sweep/map-reduce strategy:
1. Hybrid search finds ~20 potentially relevant doc_ids from xlarge chunks
2. For EACH doc: loads all large chunks, concatenates, sends full text to the LLM
3. The LLM reads 20+ full documents to extract relevant data
4. 20 LLM calls × ~30s each = ~10 minutes for a single query

Most of those 20 documents turn out to be irrelevant (`NO_RELEVANT_DATA`). The LLM wastes time reading them.

## Solution

### Universal Metadata Schema

At ingestion, extract a structured metadata JSON from every document regardless of category. The extraction prompt asks for all fields — irrelevant fields come back empty. No miscategorization risk.

```json
{
  "summary": "2-4 sentence document summary",
  "entities": ["U.S. Army Corps of Engineers", "BL Harbert International"],
  "people": ["John Smith", "Jane Doe"],
  "organizations": ["Department of Defense", "FEMA"],
  "locations": ["Huntsville, AL", "Fort Liberty, NC"],
  "dates": ["2026-01-15", "FY2026 Q3"],
  "amounts": ["$4.2M", "$12,500 monthly"],
  "identifiers": ["W912DY-26-C-0042", "Policy 4.2", "FAR 52.219-14"],
  "topics": ["construction", "military housing", "small business set-aside"],
  "procedures": ["competitive bidding", "annual performance review"],
  "action_items": ["submit report by June 30", "review compliance annually"],
  "key_facts": ["5-year period of performance", "cost-plus-fixed-fee contract"]
}
```

### Extraction at Ingestion

**When:** After parsing, before chunking (new pipeline step)

**How:** Single LLM call per document with the full document text (up to 200K chars for 256K context models). For smaller context models, extract per large-tier chunk and merge results with deduplication.

**Extraction prompt:**
```
Extract structured metadata from this document. Return ONLY valid JSON with these fields.
Leave fields as empty arrays [] if not found. Be thorough — include ALL items found.

Fields:
- summary: 2-4 sentence summary of the document's purpose and key content
- entities: named entities (companies, agencies, programs, systems)
- people: person names mentioned
- organizations: organizations, departments, agencies
- locations: cities, states, countries, facilities
- dates: dates, fiscal years, time periods mentioned
- amounts: dollar amounts, quantities, percentages
- identifiers: contract numbers, policy numbers, regulation citations, system IDs
- topics: key topics and subject areas (3-8 keywords)
- procedures: processes, procedures, rules described
- action_items: action items, requirements, deadlines
- key_facts: important specific facts, conditions, or findings

Document ({filename}):
{text}
```

**Model compatibility:**
- Gemma 4 26B-A4B: full document in single call (256K context, native JSON output)
- Smaller models (8K-32K context): per-chunk extraction with merge

### Storage

**New field on `DocumentRecord`:**
```python
metadata_tags: Mapped[dict] = mapped_column(JSON, default=dict)
```

**Migration:** Add column `metadata_tags TEXT DEFAULT '{}'` to documents table.

**Also store the summary separately** for fast access:
```python
summary: Mapped[str] = mapped_column(String, default="")
```

The summary field replaces the current approach of generating a summary and only prepending it to chunks. Now it's stored on the document record and searchable independently.

### Embedding the Summary

Create a dedicated **summary embedding** stored alongside the document's chunk embeddings in LanceDB. This is a single vector per document representing the document's overall content.

**New chunk_size_tier: `"summary"`**

At ingestion, embed the summary text as a chunk with `chunk_size_tier="summary"` and `chunk_index=-1`. The sweep strategy can then search ONLY summary-tier vectors for fast document discovery.

## Optimized Sweep Strategy

### Current Flow (Slow)
```
Question → hybrid search xlarge chunks → 20 doc_ids
→ MAP each doc (full LLM read) → 20 LLM calls → ~10 min
```

### New Flow (Fast)
```
Question → search summary embeddings → top 50 docs by summary relevance
→ Search metadata_tags for query-relevant fields (keyword match)
→ Score and rank: combine summary relevance + metadata match
→ Top 10-15 docs pass the filter
→ MAP only those docs → 10-15 LLM calls → ~5 min

Plus: metadata from filtered-out docs still available for
the synthesizer as lightweight context (no full doc read needed)
```

### Metadata-Aware Filtering

When the classifier identifies the query type, the sweep uses metadata fields relevant to that type:

| Query Intent | Primary Metadata Fields Searched |
|-------------|--------------------------------|
| "Which contracts did the army award?" | organizations, entities, identifiers, amounts |
| "What is the travel reimbursement policy?" | topics, procedures, key_facts |
| "Who attended the January budget meeting?" | people, dates, action_items |
| "What systems require FISMA compliance?" | entities, topics, identifiers |
| "Total spending on construction in FY2026?" | amounts, dates, topics, organizations |

The filtering combines:
1. **Summary embedding similarity** — semantic relevance (vector search)
2. **Metadata keyword match** — exact field matching from the query's extracted terms
3. **Threshold** — documents below a combined score are skipped for MAP

### Metadata as Lightweight Context

Documents that pass summary search but fail MAP (or are filtered out by threshold) can still contribute their metadata to the synthesizer as lightweight context:

```
"Based on metadata from 30 additional documents:
- Organizations mentioned: Army, Navy, FEMA, ...
- Total amounts found: $4.2M, $12.5M, ...
- Related contracts: W912DY-26-C-0042, ..."
```

This gives the synthesizer awareness of the broader document set without full LLM reads.

## Pipeline Changes

### Ingestion Pipeline (Updated)

```
Parse → Categorize → Extract Metadata (NEW) → Generate Summary (from metadata)
→ Chunk (4 tiers + summary tier) → Embed → Store → Extract Entities
```

The summary is now derived from the metadata extraction step rather than a separate LLM call. The metadata extraction prompt includes the summary field, so we get both in one call.

### Step: Extract Metadata

**Position:** After categorize, before chunking

**IngestStep enum:** Add `EXTRACTING_METADATA` between `CATEGORIZING` and `CHUNKING`

**Process:**
1. Send full document text + filename to the extraction LLM
2. Parse JSON response (with cleanup for markdown fences, preamble)
3. Store `metadata_tags` and `summary` on DocumentRecord
4. Use summary as the `doc_context` prepended to chunks (replaces current summary generation)

**Fallback for small context models:**
1. Split document into large-tier chunks
2. Extract metadata from each chunk
3. Merge: union all arrays, deduplicate, keep longest summary

### Queue UI

Show extraction progress: "Extracting metadata..." with the document name. On completion, show field counts: "Extracted 12 entities, 5 people, 3 amounts"

## Admin UI

### Documents Table

Add a "Metadata" column or expandable row showing the extracted metadata tags for each document. Clicking shows the full JSON.

### Metadata Re-extraction

Button on the documents page to re-extract metadata for a document (useful after updating the extraction model or prompt). Also a bulk "Re-extract All" button in Settings.

## Configuration

```python
# config.py
metadata_extraction_enabled: bool = True  # disable to skip metadata step
metadata_max_doc_length: int = 200000  # chars sent to LLM for extraction
metadata_fallback_chunk: bool = True  # per-chunk extraction for small context models
```

## Error Handling

- **LLM returns invalid JSON:** Use `json_repair` library (already a dependency via LightRAG) to attempt fix. If still invalid, store empty metadata and log warning.
- **LLM returns empty/partial fields:** Accept as-is — empty fields are valid (document may not contain that type of data).
- **Extraction timeout:** Skip metadata step, proceed with ingestion. Document works fine without metadata, just won't benefit from optimized sweep.
- **Re-extraction:** Can be triggered per-document or in bulk without re-ingesting chunks.

## Migration for Existing Documents

Existing documents without metadata can be backfilled:
1. Add "Backfill Metadata" button in Settings
2. Iterates all documents with empty `metadata_tags`
3. Loads the document's large chunks from LanceDB, reconstructs text
4. Runs extraction
5. Progress shown in Settings page

## Files Created/Modified

- Modify: `src/db/models.py` — add `metadata_tags` and `summary` fields to DocumentRecord
- Modify: `src/db/metadata.py` — add migration, update `add_document` signature
- Modify: `src/ingestion/queue.py` — add `EXTRACTING_METADATA` step, extraction logic
- Create: `src/ingestion/metadata_extractor.py` — extraction prompt, JSON parsing, merge logic
- Modify: `src/agent/strategies/map_reduce.py` — metadata-aware document filtering
- Modify: `src/agent/strategies/sweep.py` — summary embedding search
- Modify: `src/agent/graph.py` — pass metadata context to synthesizer
- Modify: `src/admin/routes.py` — metadata display, re-extraction endpoints
- Modify: `src/admin/templates/documents.html` — metadata column/expandable view
- Modify: `src/admin/templates/settings.html` — backfill button
- Modify: `src/config.py` — metadata extraction settings

## Performance Impact

**Ingestion:** +1 LLM call per document (~5-15s with Gemma 4 26B). Acceptable since ingestion is background work.

**Sweep queries:** 
- Current: 20 docs × 30s MAP = ~600s
- With metadata filter: 10 docs × 30s MAP + metadata search (~1s) = ~301s
- **~50% faster** for typical sweep queries, more for large document sets

**Storage:** ~1-5KB JSON per document + 1 summary embedding vector. Negligible.
