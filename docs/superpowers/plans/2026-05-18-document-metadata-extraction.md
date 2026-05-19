# Document Metadata Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract structured metadata from every document at ingestion time and use it to speed up sweep/map-reduce queries by filtering documents before full LLM reads.

**Architecture:** A new `metadata_extractor.py` module sends the full document text to the LLM and parses structured JSON (entities, people, organizations, locations, dates, amounts, identifiers, topics, procedures, action_items, key_facts, summary). The metadata and summary are stored on `DocumentRecord`. The summary is also embedded as a `"summary"` tier vector. The sweep/map-reduce strategy searches summary vectors first, then filters using metadata keyword matching, and only MAP-reads the narrowed document set.

**Tech Stack:** Python/FastAPI, SQLAlchemy, LanceDB, json_repair, existing LLM client

---

### Task 1: Add metadata_tags and summary fields to DocumentRecord

**Files:**
- Modify: `src/db/models.py`
- Modify: `src/db/metadata.py`
- Modify: `src/config.py`

- [ ] **Step 1: Add fields to DocumentRecord model**

In `src/db/models.py`, add two new fields to `DocumentRecord` after `source_url`:

```python
    source_url: Mapped[str] = mapped_column(String, default="")
    summary: Mapped[str] = mapped_column(String, default="")
    metadata_tags: Mapped[dict] = mapped_column(JSON, default=dict)
    uploaded_by: Mapped[str] = mapped_column(String, default="")
```

- [ ] **Step 2: Add migration in metadata.py**

In `src/db/metadata.py`, add to the `migrations` list inside `_migrate()`:

```python
                ("summary", "documents", 'TEXT DEFAULT ""'),
                ("metadata_tags", "documents", "TEXT DEFAULT '{}'"),
```

- [ ] **Step 3: Update add_document signature**

In `src/db/metadata.py`, update the `add_document` method to accept and store the new fields:

```python
    async def add_document(
        self,
        doc_id,
        filename,
        doc_type,
        acl_groups,
        chunk_count,
        uploaded_by,
        category="",
        content_hash="",
        dataset_id=0,
        source_url="",
        summary="",
        metadata_tags=None,
    ):
        record = DocumentRecord(
            doc_id=doc_id,
            dataset_id=dataset_id,
            source_url=source_url,
            summary=summary,
            metadata_tags=metadata_tags or {},
            filename=filename,
```

- [ ] **Step 4: Add config settings**

In `src/config.py`, after the `llm_concurrency` line, add:

```python
    # Metadata extraction
    metadata_extraction_enabled: bool = True  # disable to skip metadata step
    metadata_max_doc_length: int = 200000  # chars sent to LLM for extraction
```

- [ ] **Step 5: Commit**

```bash
git add src/db/models.py src/db/metadata.py src/config.py
git commit -m "feat: add metadata_tags and summary fields to DocumentRecord"
```

---

### Task 2: Create metadata extractor module

**Files:**
- Create: `src/ingestion/metadata_extractor.py`

- [ ] **Step 1: Create the extraction module**

Create `src/ingestion/metadata_extractor.py`:

```python
"""Extract structured metadata from document text via LLM."""
import json
import logging

from src.config import settings
from src.generation.llm_client import generate

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract structured metadata from this document. Return ONLY valid JSON with these fields.
Leave fields as empty arrays [] if not found. The summary field should be a string, not an array.
Be thorough — include ALL items found.

Fields:
- summary: string, 2-4 sentence summary of the document's purpose and key content
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
{text}"""

EMPTY_METADATA = {
    "summary": "",
    "entities": [],
    "people": [],
    "organizations": [],
    "locations": [],
    "dates": [],
    "amounts": [],
    "identifiers": [],
    "topics": [],
    "procedures": [],
    "action_items": [],
    "key_facts": [],
}


def extract_metadata(text: str, filename: str) -> dict:
    """Extract structured metadata from document text.

    Returns a dict with all metadata fields. On failure, returns EMPTY_METADATA.
    """
    if not settings.metadata_extraction_enabled:
        return dict(EMPTY_METADATA)

    # Truncate to configured max length
    doc_text = text[:settings.metadata_max_doc_length]

    try:
        raw = generate(
            system_prompt="You extract structured metadata from documents. Return ONLY valid JSON.",
            user_prompt=EXTRACTION_PROMPT.format(filename=filename, text=doc_text),
            temperature=0.0,
            max_tokens=4096,
        )
    except Exception as e:
        logger.warning(f"Metadata extraction LLM call failed for {filename}: {e}")
        return dict(EMPTY_METADATA)

    return _parse_metadata_response(raw, filename)


def _parse_metadata_response(raw: str, filename: str) -> dict:
    """Parse LLM response into metadata dict with fallback to json_repair."""
    import re

    # Strip markdown fences and preamble
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    # Remove any text before the first {
    brace_idx = cleaned.find("{")
    if brace_idx > 0:
        cleaned = cleaned[brace_idx:]

    # Try standard json.loads first
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: json_repair (already installed via LightRAG)
        try:
            import json_repair
            parsed = json_repair.loads(cleaned)
        except Exception as e:
            logger.warning(f"Metadata JSON parse failed for {filename}: {e}")
            logger.debug(f"Raw response: {raw[:500]}")
            return dict(EMPTY_METADATA)

    if not isinstance(parsed, dict):
        logger.warning(f"Metadata extraction returned non-dict for {filename}")
        return dict(EMPTY_METADATA)

    # Normalize: ensure all expected fields exist with correct types
    result = dict(EMPTY_METADATA)
    for key in EMPTY_METADATA:
        if key in parsed:
            if key == "summary":
                result["summary"] = str(parsed["summary"]) if parsed["summary"] else ""
            elif isinstance(parsed[key], list):
                result[key] = [str(item) for item in parsed[key] if item]
            else:
                result[key] = []
    return result


def merge_chunk_metadata(chunk_results: list[dict]) -> dict:
    """Merge metadata extracted from multiple chunks into one."""
    merged = dict(EMPTY_METADATA)
    summaries = []

    for chunk_meta in chunk_results:
        for key in EMPTY_METADATA:
            if key == "summary":
                if chunk_meta.get("summary"):
                    summaries.append(chunk_meta["summary"])
            else:
                existing = set(merged.get(key, []))
                for item in chunk_meta.get(key, []):
                    if item and item not in existing:
                        merged[key].append(item)
                        existing.add(item)

    # Use longest summary as the merged summary
    if summaries:
        merged["summary"] = max(summaries, key=len)

    return merged
```

- [ ] **Step 2: Commit**

```bash
git add src/ingestion/metadata_extractor.py
git commit -m "feat: create metadata extractor module with LLM extraction and JSON parsing"
```

---

### Task 3: Integrate metadata extraction into ingestion pipeline

**Files:**
- Modify: `src/ingestion/queue.py`

- [ ] **Step 1: Add EXTRACTING_METADATA to IngestStep enum**

In `src/ingestion/queue.py`, add the new step between CATEGORIZING and CHUNKING:

```python
class IngestStep(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    CATEGORIZING = "categorizing"
    EXTRACTING_METADATA = "extracting_metadata"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    EXTRACTING_ENTITIES = "extracting_entities"
    COMPLETE = "complete"
    FAILED = "failed"
```

- [ ] **Step 2: Replace summary generation with metadata extraction**

In `src/ingestion/queue.py`, replace the "Generate LLM document summary" block (lines ~210-222) with metadata extraction:

Replace:
```python
        # Generate LLM document summary for contextual enrichment
        self.update_step(job.job_id, IngestStep.CHUNKING, "Generating document summary")
        doc_summary = ""
        try:
            from src.generation.llm_client import generate as llm_generate
            doc_summary = await asyncio.to_thread(
                llm_generate,
                system_prompt="Summarize ALL items in this document in 2-4 sentences. List EVERY company, contract, or award mentioned — do not omit any. Include names, amounts, and dates.",
                user_prompt=parsed.text[:6000],
                temperature=0.0, max_tokens=1024,
            )
        except Exception:
            pass
```

With:
```python
        # Extract structured metadata (includes summary)
        self.update_step(job.job_id, IngestStep.EXTRACTING_METADATA, "Extracting document metadata...")
        from src.ingestion.metadata_extractor import extract_metadata
        metadata = await asyncio.to_thread(extract_metadata, parsed.text, parsed.filename)
        doc_summary = metadata.get("summary", "")
        job.metadata_tags = metadata

        # Log extraction results
        field_counts = {k: len(v) for k, v in metadata.items() if isinstance(v, list) and v}
        if field_counts:
            counts_str = ", ".join(f"{v} {k}" for k, v in field_counts.items())
            self.update_step(job.job_id, IngestStep.EXTRACTING_METADATA, f"Extracted: {counts_str}")
        else:
            self.update_step(job.job_id, IngestStep.EXTRACTING_METADATA, "No metadata extracted")
```

- [ ] **Step 3: Add metadata_tags to IngestJob dataclass**

In `src/ingestion/queue.py`, add to the `IngestJob` dataclass after `source_url`:

```python
    source_url: str = ""
    metadata_tags: dict = field(default_factory=dict)
    auto_categorize: bool = True
```

- [ ] **Step 4: Pass metadata to add_document**

In `src/ingestion/queue.py`, update the `add_document` call (around line 262) to pass the new fields:

```python
        await metadata_store.add_document(
            doc_id=doc_id, filename=parsed.filename, doc_type=parsed.doc_type,
            acl_groups=job.acl_groups, chunk_count=total_chunks,
            uploaded_by=job.uploaded_by, category=category,
            content_hash=content_hash, dataset_id=job.dataset_id,
            source_url=job.source_url,
            summary=doc_summary,
            metadata_tags=job.metadata_tags,
        )
```

- [ ] **Step 5: Add summary embedding as a "summary" tier chunk**

In `src/ingestion/queue.py`, after the CHUNK_TIERS loop (after all tiers are embedded and stored), add the summary embedding:

```python
        # Embed the summary as a dedicated "summary" tier for fast document discovery
        if doc_summary:
            self.update_step(job.job_id, IngestStep.EMBEDDING, "Embedding document summary")
            summary_text = f"{doc_context}\n{doc_summary}"
            summary_vector = await asyncio.to_thread(embed_texts, [summary_text])
            if summary_vector:
                from src.retrieval.models import ChunkMetadata
                summary_meta = ChunkMetadata(
                    doc_id=doc_id, filename=parsed.filename, doc_type=parsed.doc_type,
                    chunk_index=-1, start_char=0, acl_groups=job.acl_groups,
                    category=category, chunk_size_tier="summary",
                )
                await asyncio.to_thread(
                    vector_store.upsert,
                    texts=[summary_text], vectors=summary_vector, metadatas=[summary_meta],
                )
```

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/queue.py
git commit -m "feat: integrate metadata extraction into ingestion pipeline with summary embeddings"
```

---

### Task 4: Optimize sweep/map-reduce with metadata filtering

**Files:**
- Modify: `src/agent/strategies/map_reduce.py`
- Modify: `src/agent/strategies/sweep.py`

- [ ] **Step 1: Add metadata filtering to map_reduce.py**

In `src/agent/strategies/map_reduce.py`, replace the document discovery section (the `else` branch after date_filter_docs check) with a two-phase approach — summary search then metadata filter:

Replace the else block:
```python
    else:
        initial_results = vector_store.hybrid_search(
            vector=query_vector, text_query=question,
            user_groups=user_groups, top_k=top_k, tier="xlarge", doc_ids=doc_ids,
        )
        relevant_doc_ids = list({chunk.metadata.doc_id for chunk in initial_results})
        # Log which documents were discovered
        doc_filenames = {c.metadata.doc_id: c.metadata.filename for c in initial_results}
        logger.info(f"Map-reduce: found {len(relevant_doc_ids)} relevant documents from {len(initial_results)} xlarge chunks")
        for did in relevant_doc_ids:
            logger.info(f"  - {doc_filenames.get(did, did)}")
```

With:
```python
    else:
        # Phase 1: Search summary embeddings for fast document discovery
        summary_results = vector_store.search(
            vector=query_vector, user_groups=user_groups,
            top_k=top_k, tier="summary", doc_ids=doc_ids,
        )
        # Fallback to xlarge if no summary embeddings exist yet
        if not summary_results:
            summary_results = vector_store.hybrid_search(
                vector=query_vector, text_query=question,
                user_groups=user_groups, top_k=top_k, tier="xlarge", doc_ids=doc_ids,
            )
            logger.info("Map-reduce: no summary embeddings found, falling back to xlarge search")

        candidate_doc_ids = list({c.metadata.doc_id for c in summary_results})
        doc_filenames = {c.metadata.doc_id: c.metadata.filename for c in summary_results}
        doc_scores = {c.metadata.doc_id: c.score for c in summary_results}
        logger.info(f"Map-reduce: {len(candidate_doc_ids)} candidate docs from summary/xlarge search")

        # Phase 2: Filter using metadata keyword matching
        from src.ingestion.metadata_extractor import EMPTY_METADATA
        metadata_store = None
        try:
            from src.api.routes_ingest import get_metadata_store
            metadata_store = get_metadata_store()
        except Exception:
            pass

        relevant_doc_ids = candidate_doc_ids  # default: no filtering if metadata unavailable
        if metadata_store:
            q_lower = question.lower()
            scored_docs = []
            for did in candidate_doc_ids:
                doc_rec = await metadata_store.get_document(did)
                meta = getattr(doc_rec, 'metadata_tags', {}) or {} if doc_rec else {}
                if not meta:
                    scored_docs.append((did, doc_scores.get(did, 0)))
                    continue
                # Score: count how many metadata values match query terms
                meta_score = 0
                for field, values in meta.items():
                    if field == "summary" or not isinstance(values, list):
                        continue
                    for val in values:
                        if val and val.lower() in q_lower:
                            meta_score += 2
                        elif val and any(word in q_lower for word in val.lower().split() if len(word) > 3):
                            meta_score += 1
                combined = doc_scores.get(did, 0) + (meta_score * 0.1)
                scored_docs.append((did, combined))

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            relevant_doc_ids = [did for did, _ in scored_docs[:30]]

        logger.info(f"Map-reduce: {len(relevant_doc_ids)} docs after metadata filtering (from {len(candidate_doc_ids)} candidates)")
        for did in relevant_doc_ids:
            logger.info(f"  - {doc_filenames.get(did, did)}")
```

- [ ] **Step 2: Update sweep.py to use summary tier**

In `src/agent/strategies/sweep.py`, update the else block in `retrieve_sweep` to search summary embeddings first:

Replace:
```python
    else:
        # General sweep: find relevant documents via search
        initial_results = vector_store.hybrid_search(
            vector=query_vector, text_query=question,
            user_groups=user_groups, top_k=top_k, tier="xlarge", doc_ids=doc_ids,
        )
        relevant_doc_ids = list({chunk.metadata.doc_id for chunk in initial_results})
        logger.info(f"Sweep: found {len(relevant_doc_ids)} relevant documents from xlarge search")
```

With:
```python
    else:
        # Search summary embeddings first, fall back to xlarge
        initial_results = vector_store.search(
            vector=query_vector, user_groups=user_groups,
            top_k=top_k, tier="summary", doc_ids=doc_ids,
        )
        if not initial_results:
            initial_results = vector_store.hybrid_search(
                vector=query_vector, text_query=question,
                user_groups=user_groups, top_k=top_k, tier="xlarge", doc_ids=doc_ids,
            )
            logger.info("Sweep: no summary embeddings, falling back to xlarge")
        relevant_doc_ids = list({chunk.metadata.doc_id for chunk in initial_results})
        logger.info(f"Sweep: found {len(relevant_doc_ids)} relevant documents from summary search")
```

- [ ] **Step 3: Commit**

```bash
git add src/agent/strategies/map_reduce.py src/agent/strategies/sweep.py
git commit -m "feat: optimize sweep/map-reduce with summary embeddings and metadata filtering"
```

---

### Task 5: Add metadata context to synthesizer

**Files:**
- Modify: `src/agent/graph.py`

- [ ] **Step 1: Pass metadata from non-MAP'd documents as lightweight context**

In `src/agent/graph.py`, inside the `retrieve` function, after the SWEEP branch merges sweep + map-reduce results, add metadata context from discovered-but-not-MAP'd documents:

After the existing merge block for SWEEP, add:

```python
            # Add lightweight metadata context from documents not fully MAP'd
            try:
                from src.api.routes_ingest import get_metadata_store
                _ms = get_metadata_store()
                sweep_doc_ids = {c.metadata.doc_id for c in sweep_result.get("retrieved_chunks", [])}
                mr_doc_ids = set()
                for c in mr_result.get("retrieved_chunks", []):
                    if c.metadata.doc_id != "map-reduce":
                        mr_doc_ids.add(c.metadata.doc_id)
                # Documents discovered by sweep but not in map-reduce results
                extra_doc_ids = sweep_doc_ids - mr_doc_ids - {"map-reduce", "knowledge-graph"}
                if extra_doc_ids:
                    meta_parts = []
                    for did in list(extra_doc_ids)[:20]:
                        doc_rec = await _ms.get_document(did)
                        if doc_rec and getattr(doc_rec, 'metadata_tags', None):
                            meta = doc_rec.metadata_tags
                            parts = []
                            for field in ["entities", "organizations", "amounts", "identifiers", "topics"]:
                                vals = meta.get(field, [])
                                if vals:
                                    parts.append(f"{field}: {', '.join(vals[:5])}")
                            if parts:
                                meta_parts.append(f"[{doc_rec.filename}]: {'; '.join(parts)}")
                    if meta_parts:
                        meta_chunk = RetrievedChunk(
                            text=f"Additional document metadata ({len(meta_parts)} docs not fully analyzed):\n" + "\n".join(meta_parts),
                            score=0.3,
                            metadata=ChunkMetadata(
                                doc_id="metadata-context", filename="metadata_context",
                                doc_type="metadata", chunk_index=0, start_char=0, acl_groups=["ALL"],
                            ),
                        )
                        merged_chunks.append(meta_chunk)
                        retrieve_logger.info(f"Added metadata context from {len(meta_parts)} additional documents")
            except Exception as e:
                retrieve_logger.debug(f"Metadata context enrichment skipped: {e}")
```

- [ ] **Step 2: Commit**

```bash
git add src/agent/graph.py
git commit -m "feat: add lightweight metadata context from non-MAP'd documents to synthesizer"
```

---

### Task 6: Add metadata display and re-extraction to admin UI

**Files:**
- Modify: `src/admin/routes.py`
- Modify: `src/admin/templates/settings.html`

- [ ] **Step 1: Add re-extraction endpoint**

In `src/admin/routes.py`, after the `purge_knowledge_graph` endpoint, add:

```python
@router.post("/api/settings/backfill-metadata")
async def backfill_metadata():
    """Re-extract metadata for all documents missing metadata_tags."""
    import asyncio

    store = get_metadata_store()
    docs = await store.list_documents()
    missing = [d for d in docs if not getattr(d, 'metadata_tags', None)]

    if not missing:
        return HTMLResponse('<span class="status-ok">All documents already have metadata.</span>')

    async def _backfill():
        from src.ingestion.metadata_extractor import extract_metadata
        from src.ingestion.embedder import embed_texts
        vs = get_vector_store()
        from src.retrieval.models import ChunkMetadata

        for i, doc in enumerate(missing):
            try:
                # Reconstruct text from large chunks
                chunks = vs.get_chunks_by_doc(doc.doc_id, limit=200, tier="large")
                if not chunks:
                    chunks = vs.get_chunks_by_doc(doc.doc_id, limit=200)
                if not chunks:
                    continue
                text = "\n\n".join(c.text for c in chunks)

                metadata = await asyncio.to_thread(extract_metadata, text, doc.filename)
                summary = metadata.get("summary", "")

                await store.update_document(doc.doc_id, summary=summary, metadata_tags=metadata)

                # Embed and store summary vector
                if summary:
                    doc_context = f"Document: {doc.filename} (type: {doc.doc_type}, category: {doc.category})\nSummary: {summary}"
                    vectors = await asyncio.to_thread(embed_texts, [doc_context])
                    if vectors:
                        meta = ChunkMetadata(
                            doc_id=doc.doc_id, filename=doc.filename, doc_type=doc.doc_type,
                            chunk_index=-1, start_char=0, acl_groups=doc.acl_groups,
                            category=doc.category, chunk_size_tier="summary",
                        )
                        await asyncio.to_thread(vs.upsert, texts=[doc_context], vectors=vectors, metadatas=[meta])
            except Exception as e:
                logging.getLogger(__name__).warning(f"Backfill failed for {doc.filename}: {e}")

    asyncio.create_task(_backfill())
    return HTMLResponse(f'<span style="color:#2563eb;">Backfilling metadata for {len(missing)} documents in background...</span>')
```

- [ ] **Step 2: Add backfill button to settings page**

In `src/admin/templates/settings.html`, inside the "Knowledge Graph" settings section (after the purge button), add:

```html
    <div style="margin-top:1rem;">
        <button type="button" hx-post="/admin/api/settings/backfill-metadata" hx-target="#metadata-status" hx-swap="innerHTML">Backfill Document Metadata</button>
        <span id="metadata-status" style="margin-left:0.5rem;"></span>
        <p class="section-desc" style="margin-top:0.25rem;">Extract metadata for documents ingested before this feature was added. Runs in background.</p>
    </div>
```

- [ ] **Step 3: Add metadata_extraction_enabled and metadata_max_doc_length to settings form and save handler**

In the settings save endpoint, add the new params and persist them:

Add to function signature:
```python
    metadata_extraction_enabled: bool = Form(True),
    metadata_max_doc_length: int = Form(200000),
```

Add to the in-memory update section:
```python
    settings.metadata_extraction_enabled = metadata_extraction_enabled
    settings.metadata_max_doc_length = metadata_max_doc_length
```

Add to the env_lines section:
```python
    env_lines["METADATA_EXTRACTION_ENABLED"] = str(settings.metadata_extraction_enabled).lower()
    env_lines["METADATA_MAX_DOC_LENGTH"] = str(settings.metadata_max_doc_length)
```

Add to the settings template, in the Concurrency section or as a new section:
```html
    <div style="margin-top:1rem;">
        <h3 style="margin-bottom:0.5rem;">Metadata Extraction</h3>
        <div class="form-row">
            <div class="form-group">
                <label for="metadata_extraction_enabled">Enable Metadata Extraction</label>
                <select id="metadata_extraction_enabled" name="metadata_extraction_enabled">
                    <option value="true" {{ 'selected' if settings.metadata_extraction_enabled }}>Enabled</option>
                    <option value="false" {{ 'selected' if not settings.metadata_extraction_enabled }}>Disabled</option>
                </select>
            </div>
            <div class="form-group">
                <label for="metadata_max_doc_length">Max Document Length (chars)</label>
                <input type="number" id="metadata_max_doc_length" name="metadata_max_doc_length" value="{{ settings.metadata_max_doc_length }}" min="1000" max="500000" style="max-width:120px;">
                <span style="font-size:0.8rem; color:#6b7280;">Chars sent to LLM for extraction</span>
            </div>
        </div>
    </div>
```

- [ ] **Step 4: Commit**

```bash
git add src/admin/routes.py src/admin/templates/settings.html
git commit -m "feat: add metadata backfill endpoint and settings UI"
```

---

### Task 7: Update queue status UI to show metadata step

**Files:**
- Modify: `src/admin/routes.py`

- [ ] **Step 1: Update the step labels in the playground and queue status**

In `src/admin/routes.py`, find the `step_labels` dict in the playground `run_query` function and add the new step:

```python
            step_labels = {"cache_check": "Check Cache", "classify": "Classify Query", "retrieve": "Retrieve Documents", "enrich": "Knowledge Graph Enrichment", "synthesize": "Generate Answer"}
```

This doesn't need changing since metadata extraction happens at ingestion time (in the queue), not at query time. But the **queue status HTML builder** needs to recognize the new step. Find the queue status endpoint and ensure `extracting_metadata` renders properly.

In the playground STEPS JavaScript array (`src/admin/templates/playground.html`), no changes needed — the metadata step is ingestion-only, not query-time.

In the queue status rendering (`src/admin/routes.py`, the `queue_status` function), the existing code already handles unknown steps with a blue label:
```python
        else:
            status = f'<span style="color: #2563eb; font-weight: 600;">{job.step}</span>'
```

So `extracting_metadata` will show as a blue "extracting_metadata" label automatically. For a cleaner display, add it to the step name formatting in the queue status. Find the line that formats `job.step` and add a display name:

After the queue status `rows` variable is built, no code change needed — the step name displays as-is.

- [ ] **Step 2: Commit**

```bash
git commit --allow-empty -m "chore: verify queue UI handles extracting_metadata step"
```

---

### Task 8: Final integration test and documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Add a section about metadata extraction under the "Ingestion Pipeline" section:

```markdown
## Document Metadata

Every document gets structured metadata extracted at ingestion time via a single LLM call:

- **summary** -- 2-4 sentence document overview
- **entities, people, organizations, locations** -- named entities
- **dates, amounts, identifiers** -- structured data points
- **topics, procedures, action_items, key_facts** -- semantic content

This metadata is used to speed up sweep queries — instead of the LLM reading every document, the system first searches document summaries and metadata to narrow the list, then only reads the truly relevant ones. Existing documents can be backfilled from Settings.
```

- [ ] **Step 2: Commit and push**

```bash
git add README.md
git commit -m "docs: add document metadata extraction to README"
git push origin master
```
