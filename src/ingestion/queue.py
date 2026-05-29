from __future__ import annotations
# src/ingestion/queue.py
from __future__ import annotations
"""Ingestion job queue with step-level status tracking."""
import asyncio
import logging
import time
import threading
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_structured_pdf(doc_type: str) -> bool:
    return doc_type == "pdf"


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


@dataclass
class IngestJob:
    job_id: str
    filename: str
    file_path: str
    acl_groups: list[str]
    uploaded_by: str
    category: str = ""
    dataset_id: int = 0
    source_url: str = ""
    metadata_tags: dict = field(default_factory=dict)
    auto_categorize: bool = True
    build_graph: bool = True
    step: IngestStep = IngestStep.QUEUED
    progress: str = ""
    doc_id: str = ""
    chunk_count: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    error: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0


class IngestQueue:
    def __init__(self):
        self._jobs: dict[str, IngestJob] = {}
        self._queue: asyncio.Queue | None = None
        self._worker_running = False
        # KG extraction runs without locks — counts are approximate when parallel

    def enqueue(self, filename: str, file_path: str, acl_groups: list[str],
                uploaded_by: str, category: str = "", dataset_id: int = 0,
                source_url: str = "",
                auto_categorize: bool = True, build_graph: bool = True) -> str:
        job_id = str(uuid.uuid4())[:8]
        job = IngestJob(
            job_id=job_id, filename=filename, file_path=file_path,
            acl_groups=acl_groups, uploaded_by=uploaded_by,
            category=category, dataset_id=dataset_id, source_url=source_url,
            auto_categorize=auto_categorize, build_graph=build_graph,
        )
        self._jobs[job_id] = job
        if self._queue:
            self._queue.put_nowait(job_id)
        return job_id

    def get_job(self, job_id: str) -> IngestJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[IngestJob]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def has_active_jobs(self) -> bool:
        """True if any job is queued or mid-processing (not in a terminal state).

        Used to block destructive operations (e.g. purging the knowledge graph)
        while an in-flight ingestion could still be writing to it.
        """
        terminal = (IngestStep.COMPLETE, IngestStep.FAILED)
        return any(job.step not in terminal for job in self._jobs.values())

    def update_step(self, job_id: str, step: IngestStep, progress: str = ""):
        job = self._jobs.get(job_id)
        if job:
            job.step = step
            job.progress = progress

    def complete_job(self, job_id: str, doc_id: str, chunk_count: int, entity_count: int = 0, relationship_count: int = 0):
        job = self._jobs.get(job_id)
        if job:
            job.step = IngestStep.COMPLETE
            job.doc_id = doc_id
            job.chunk_count = chunk_count
            job.entity_count = entity_count
            job.relationship_count = relationship_count
            job.completed_at = time.time()

    def fail_job(self, job_id: str, error: str):
        job = self._jobs.get(job_id)
        if job:
            job.step = IngestStep.FAILED
            job.error = error
            job.completed_at = time.time()

    max_parallel: int = 3  # max concurrent ingestion jobs

    async def start_worker(self, vector_store, metadata_store):
        """Start the background worker that processes the queue."""
        if self._worker_running:
            return
        self._queue = asyncio.Queue()
        self._worker_running = True

        # Re-queue any jobs that were queued before worker started
        for job in self._jobs.values():
            if job.step == IngestStep.QUEUED:
                self._queue.put_nowait(job.job_id)

        # Start multiple worker tasks for parallel processing
        from src.config import settings
        self.max_parallel = settings.max_parallel_ingestion
        for i in range(self.max_parallel):
            asyncio.create_task(self._worker_loop(vector_store, metadata_store))

    async def _worker_loop(self, vector_store, metadata_store):
        import traceback
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if not job:
                continue
            try:
                await self._process_job(job, vector_store, metadata_store)
            except Exception as e:
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                logger.error(f"Ingestion failed for {job.filename} at step {job.step}: {error_msg}")
                self.fail_job(job_id, error_msg)
                # Roll back partial writes so a failed job can't orphan chunks.
                # Ingestion writes vectors before committing metadata, so a
                # mid-flight failure would otherwise leave LanceDB chunks with
                # no owning document. Deleting by this job's doc_id is bounded
                # to exactly this job (a no-op if nothing was written yet).
                await self._cleanup_failed_job(job, vector_store, metadata_store)
                # Write to file for debugging
                with open('/tmp/ingest_errors.log', 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Job: {job.filename}\n")
                    f.write(f"Step: {job.step}\n")
                    f.write(f"Progress: {job.progress}\n")
                    f.write(f"Error:\n{error_msg}\n")
            self._queue.task_done()

    async def _cleanup_failed_job(self, job: IngestJob, vector_store, metadata_store):
        """Remove any partial writes left by a failed job, keyed on its doc_id.

        Bounded to the failed job's own doc_id, so it is safe even if the rest
        of the store is healthy. Both deletes are no-ops when nothing was
        written for the doc_id yet (e.g. early failures).
        """
        doc_id = job.doc_id
        if not doc_id:
            return
        try:
            vector_store.delete_by_doc_id(doc_id)
        except Exception as ce:
            logger.warning(f"Partial-chunk cleanup failed for {job.filename} ({doc_id}): {ce}")
        try:
            await metadata_store.delete_document(doc_id)
        except Exception as ce:
            logger.warning(f"Partial-metadata cleanup failed for {job.filename} ({doc_id}): {ce}")
        try:
            from src.ingestion.tabular_ingest import cleanup_spreadsheet_tables
            await cleanup_spreadsheet_tables(doc_id, metadata_store)
        except Exception as ce:
            logger.warning(f"Partial-tabular cleanup failed for {job.filename} ({doc_id}): {ce}")

    async def _process_job(self, job: IngestJob, vector_store, metadata_store):
        import asyncio
        from src.ingestion.parser import parse_document
        from src.ingestion.chunker import chunk_text
        from src.ingestion.embedder import embed_texts
        from src.ingestion.tabular_ingest import ingest_structured_sheets, ingest_grids, SPREADSHEET_DOC_TYPES
        from src.ingestion.tabular_chunker import sheets_needing_text, build_tier_chunks
        from src.ingestion.pdf_extract import extract_pdf
        from src.knowledge.categorizer import categorize_document
        from src.retrieval.models import ChunkMetadata

        doc_id = str(uuid.uuid4())
        job.doc_id = doc_id  # record early so a failed job's partial writes can be cleaned up
        file_path = Path(job.file_path)

        # Duplicate check: hash file content before doing any work
        import hashlib
        content_bytes = file_path.read_bytes()
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # Check by content hash
        existing = await metadata_store.find_by_content_hash(content_hash)
        if existing:
            self.fail_job(job.job_id, f"Duplicate: identical content already ingested as '{existing.filename}' (doc_id: {existing.doc_id[:8]}...)")
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass
            return

        # Also check by filename (catches pre-hash duplicates)
        all_docs = await metadata_store.list_documents(None)
        filename_match = [d for d in all_docs if d.filename == job.filename]
        if filename_match:
            self.fail_job(job.job_id, f"Duplicate: a document named '{job.filename}' already exists (doc_id: {filename_match[0].doc_id[:8]}...). Delete it first to re-ingest.")
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass
            return

        # Step 1: Parse (fast, ok on event loop)
        self.update_step(job.job_id, IngestStep.PARSING, f"Parsing {job.filename}")
        parsed = await asyncio.to_thread(parse_document, file_path)
        parsed.filename = job.filename

        # Step 2: Categorize (LLM call — run in thread)
        category = job.category
        if not category and job.auto_categorize:
            self.update_step(job.job_id, IngestStep.CATEGORIZING, "Auto-categorizing with LLM")
            cat_result = await asyncio.to_thread(
                categorize_document,
                filename=parsed.filename, doc_type=parsed.doc_type,
                text_preview=parsed.text[:500], metadata_store=metadata_store,
            )
            if cat_result.is_new:
                await metadata_store.add_proposal(
                    proposed_name=cat_result.category, proposed_description=cat_result.description,
                    proposed_acl_groups=cat_result.suggested_acl_groups,
                    proposed_keywords=cat_result.suggested_keywords,
                    proposed_grs=cat_result.suggested_grs, proposed_by="auto-categorizer",
                )
                category = "uncategorized"
            else:
                category = cat_result.category
            job.category = category  # update job so queue UI shows it

        # Inherit default ACL from category if none provided
        if not job.acl_groups and category and category != "uncategorized":
            cat_record = await metadata_store.get_category(category)
            if cat_record and cat_record.acl_groups:
                job.acl_groups = cat_record.acl_groups

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

        # Step 3-5: Multi-pass chunking, embedding, and storage
        CHUNK_TIERS = [
            ("small", 1024, 100),
            ("medium", 2048, 200),
            ("large", 4096, 400),
            ("xlarge", 8192, 800),
        ]
        doc_context = f"Document: {parsed.filename} (type: {parsed.doc_type}, category: {category})"
        if doc_summary:
            doc_context += f"\nSummary: {doc_summary}"
        total_chunks = 0
        chunks = []  # ensure defined even if all sheets de-dup to zero text chunks

        # Smaller batch sizes for larger tiers to prevent OOM kills
        TIER_BATCH_SIZES = {"small": 64, "medium": 32, "large": 16, "xlarge": 8}

        # Structured handling for spreadsheets: clean sheets -> DuckDB + schema +
        # row narratives; messy sheets -> deterministic region narratives. Returns
        # which clean sheets fully succeeded so we de-dup their full text below.
        # Mirrors the sync pipeline so both ingestion paths behave identically.
        is_spreadsheet = parsed.doc_type in SPREADSHEET_DOC_TYPES
        is_pdf = _is_structured_pdf(parsed.doc_type)
        text_sheets = None
        pdf_prose = None
        if is_spreadsheet:
            self.update_step(job.job_id, IngestStep.STORING, "Structured spreadsheet ingest (DuckDB + narratives)")
            grids, classifications, ingested = await ingest_structured_sheets(
                file_path, doc_id, parsed.filename, parsed.doc_type,
                job.acl_groups, category, vector_store, metadata_store,
            )
            text_sheets = sheets_needing_text(grids, classifications, ingested)
        elif is_pdf:
            self.update_step(job.job_id, IngestStep.STORING, "Structured PDF ingest (tables -> DuckDB + narratives)")
            try:
                extracted = await asyncio.to_thread(extract_pdf, Path(file_path))
                await ingest_grids(
                    extracted.table_grids, doc_id, parsed.filename, parsed.doc_type,
                    job.acl_groups, category, vector_store, metadata_store,
                )
                pdf_prose = "\n\n".join(b.text for b in extracted.prose_blocks)
            except Exception as e:
                logger.warning(f"PDF structured extract failed for {parsed.filename}, "
                               f"falling back to flat text: {e}")
                is_pdf = False

        for tier_name, tier_size, tier_overlap in CHUNK_TIERS:
            self.update_step(job.job_id, IngestStep.CHUNKING, f"Chunking at {tier_name} ({tier_size} chars)")
            if is_spreadsheet:
                # Structure-aware, row-atomic chunks for messy + failed-clean sheets
                # only. Clean sheets already in the structured store contribute none.
                tier_chunks = build_tier_chunks(text_sheets, chunk_size=tier_size)
            elif is_pdf:
                tier_chunks = chunk_text(pdf_prose or "", chunk_size=tier_size, chunk_overlap=tier_overlap)
            else:
                tier_chunks = chunk_text(parsed.text, chunk_size=tier_size, chunk_overlap=tier_overlap)

            self.update_step(job.job_id, IngestStep.EMBEDDING, f"Embedding {len(tier_chunks)} {tier_name} chunks")
            texts = [f"{doc_context}\n\n{c.text}" for c in tier_chunks]
            metadatas = [
                ChunkMetadata(
                    doc_id=doc_id, filename=parsed.filename, doc_type=parsed.doc_type,
                    chunk_index=c.index, start_char=c.start_char,
                    acl_groups=job.acl_groups, category=category,
                    chunk_size_tier=tier_name,
                )
                for c in tier_chunks
            ]
            batch_size = TIER_BATCH_SIZES.get(tier_name, 32)
            vectors = await asyncio.to_thread(embed_texts, texts, "passage", batch_size) if texts else []

            self.update_step(job.job_id, IngestStep.STORING, f"Storing {tier_name} chunks")
            if vectors:
                await asyncio.to_thread(vector_store.upsert, texts=texts, vectors=vectors, metadatas=metadatas)

            if tier_name == "medium":
                total_chunks = len(tier_chunks)
                chunks = tier_chunks  # use medium tier for entity extraction
                job.chunk_count = total_chunks  # show count in UI immediately

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

        await metadata_store.add_document(
            doc_id=doc_id, filename=parsed.filename, doc_type=parsed.doc_type,
            acl_groups=job.acl_groups, chunk_count=total_chunks,
            uploaded_by=job.uploaded_by, category=category,
            content_hash=content_hash, dataset_id=job.dataset_id,
            source_url=job.source_url, summary=doc_summary, metadata_tags=job.metadata_tags,
        )
        if category and category != "uncategorized":
            existing = await metadata_store.get_category(category)
            if not existing:
                await metadata_store.add_category(
                    name=category, description="", acl_groups=job.acl_groups, routing_keywords=[],
                )

        # (Spreadsheet structured ingest + de-dup happened in the chunking loop above.)

        # Step 6: Build knowledge graph via LightRAG. Skipped for spreadsheets —
        # they are fully covered by the structured/tabular store, and KG
        # extraction over flattened numeric tables is costly and near-empty.
        if is_spreadsheet:
            self.update_step(job.job_id, IngestStep.EXTRACTING_ENTITIES,
                "Skipped (structured data — knowledge graph not applicable)")
        elif job.build_graph:
            self.update_step(job.job_id, IngestStep.EXTRACTING_ENTITIES, "Building knowledge graph...")
            from src.knowledge.graph_rag import insert_document as lightrag_insert, get_graph_counts

            MAX_RETRIES = 3
            TIMEOUT_SECS = 600  # 10 minutes per attempt
            import logging as _log
            _kg_logger = _log.getLogger(__name__)

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    nodes_before, edges_before = await get_graph_counts()

                    if attempt > 1:
                        self.update_step(job.job_id, IngestStep.EXTRACTING_ENTITIES,
                            f"Knowledge graph retry {attempt}/{MAX_RETRIES}...")

                    await asyncio.wait_for(
                        lightrag_insert(parsed.text, doc_id=doc_id, filename=parsed.filename),
                        timeout=TIMEOUT_SECS,
                    )

                    nodes_after, edges_after = await get_graph_counts()
                    job.entity_count = max(0, nodes_after - nodes_before)
                    job.relationship_count = max(0, edges_after - edges_before)

                    self.update_step(job.job_id, IngestStep.EXTRACTING_ENTITIES,
                        f"Knowledge graph complete ({job.entity_count} entities, {job.relationship_count} relationships)")
                    break  # success
                except asyncio.TimeoutError:
                    _kg_logger.warning(f"KG extraction timed out for {parsed.filename} (attempt {attempt}/{MAX_RETRIES})")
                    if attempt == MAX_RETRIES:
                        self.update_step(job.job_id, IngestStep.EXTRACTING_ENTITIES,
                            f"Knowledge graph timed out after {MAX_RETRIES} attempts")
                except Exception as e:
                    _kg_logger.warning(f"KG extraction failed for {parsed.filename} (attempt {attempt}/{MAX_RETRIES}): {e}")
                    if attempt == MAX_RETRIES:
                        self.update_step(job.job_id, IngestStep.EXTRACTING_ENTITIES,
                            f"Knowledge graph failed: {str(e)[:100]}")

            self.update_step(job.job_id, IngestStep.EXTRACTING_ENTITIES, "Knowledge graph complete")
        else:
            self.update_step(job.job_id, IngestStep.EXTRACTING_ENTITIES, "Skipped (knowledge graph disabled)")

        # Done
        self.complete_job(job.job_id, doc_id=doc_id, chunk_count=len(chunks),
                          entity_count=job.entity_count, relationship_count=job.relationship_count)

        # Cleanup temp file
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass


# Singleton
ingest_queue = IngestQueue()
