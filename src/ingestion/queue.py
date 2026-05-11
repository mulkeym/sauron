# src/ingestion/queue.py
from __future__ import annotations
"""Ingestion job queue with step-level status tracking."""
import asyncio
import time
import threading
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class IngestStep(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    CATEGORIZING = "categorizing"
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
    auto_categorize: bool = True
    step: IngestStep = IngestStep.QUEUED
    progress: str = ""
    doc_id: str = ""
    chunk_count: int = 0
    error: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0


class IngestQueue:
    def __init__(self):
        self._jobs: dict[str, IngestJob] = {}
        self._queue: asyncio.Queue | None = None
        self._worker_running = False

    def enqueue(self, filename: str, file_path: str, acl_groups: list[str],
                uploaded_by: str, category: str = "", auto_categorize: bool = True) -> str:
        job_id = str(uuid.uuid4())[:8]
        job = IngestJob(
            job_id=job_id, filename=filename, file_path=file_path,
            acl_groups=acl_groups, uploaded_by=uploaded_by,
            category=category, auto_categorize=auto_categorize,
        )
        self._jobs[job_id] = job
        if self._queue:
            self._queue.put_nowait(job_id)
        return job_id

    def get_job(self, job_id: str) -> IngestJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[IngestJob]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def update_step(self, job_id: str, step: IngestStep, progress: str = ""):
        job = self._jobs.get(job_id)
        if job:
            job.step = step
            job.progress = progress

    def complete_job(self, job_id: str, doc_id: str, chunk_count: int):
        job = self._jobs.get(job_id)
        if job:
            job.step = IngestStep.COMPLETE
            job.doc_id = doc_id
            job.chunk_count = chunk_count
            job.completed_at = time.time()

    def fail_job(self, job_id: str, error: str):
        job = self._jobs.get(job_id)
        if job:
            job.step = IngestStep.FAILED
            job.error = error
            job.completed_at = time.time()

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
                self.fail_job(job_id, error_msg)
                # Write to file for debugging
                with open('/tmp/ingest_errors.log', 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Job: {job.filename}\n")
                    f.write(f"Step: {job.step}\n")
                    f.write(f"Progress: {job.progress}\n")
                    f.write(f"Error:\n{error_msg}\n")
            self._queue.task_done()

    async def _process_job(self, job: IngestJob, vector_store, metadata_store):
        import asyncio
        from src.ingestion.parser import parse_document
        from src.ingestion.chunker import chunk_text
        from src.ingestion.embedder import embed_texts
        from src.knowledge.categorizer import categorize_document
        from src.knowledge.extractor import extract_entities
        from src.retrieval.models import ChunkMetadata

        doc_id = str(uuid.uuid4())
        file_path = Path(job.file_path)

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
                    proposed_keywords=cat_result.suggested_keywords, proposed_by="auto-categorizer",
                )
                category = "uncategorized"
            else:
                category = cat_result.category

        # Inherit default ACL from category if none provided
        if not job.acl_groups and category and category != "uncategorized":
            cat_record = await metadata_store.get_category(category)
            if cat_record and cat_record.acl_groups:
                job.acl_groups = cat_record.acl_groups

        # Step 3: Chunk (fast, ok on event loop)
        self.update_step(job.job_id, IngestStep.CHUNKING, "Splitting into chunks")
        chunks = chunk_text(parsed.text, chunk_size=1024, chunk_overlap=100)

        # Step 4: Embed (API call — run in thread)
        self.update_step(job.job_id, IngestStep.EMBEDDING, f"Embedding {len(chunks)} chunks")
        texts = [c.text for c in chunks]
        metadatas = [
            ChunkMetadata(
                doc_id=doc_id, filename=parsed.filename, doc_type=parsed.doc_type,
                chunk_index=c.index, start_char=c.start_char,
                acl_groups=job.acl_groups, category=category,
            )
            for c in chunks
        ]
        vectors = await asyncio.to_thread(embed_texts, texts) if texts else []

        # Step 5: Store (API call — run in thread)
        self.update_step(job.job_id, IngestStep.STORING, "Storing in vector DB")
        if vectors:
            await asyncio.to_thread(vector_store.upsert, texts=texts, vectors=vectors, metadatas=metadatas)
        await metadata_store.add_document(
            doc_id=doc_id, filename=parsed.filename, doc_type=parsed.doc_type,
            acl_groups=job.acl_groups, chunk_count=len(chunks),
            uploaded_by=job.uploaded_by, category=category,
        )
        if category and category != "uncategorized":
            existing = await metadata_store.get_category(category)
            if not existing:
                await metadata_store.add_category(
                    name=category, description="", acl_groups=job.acl_groups, routing_keywords=[],
                )

        # Step 6: Extract entities (LLM call per chunk — run in thread)
        for i, chunk in enumerate(chunks):
            self.update_step(job.job_id, IngestStep.EXTRACTING_ENTITIES, f"Extracting entities from chunk {i+1}/{len(chunks)}")
            extraction = await asyncio.to_thread(extract_entities, chunk.text)
            entity_id_map = {}
            for ent in extraction.entities:
                if not isinstance(ent, dict) or "name" not in ent or "type" not in ent:
                    continue
                eid = await metadata_store.add_entity(name=ent["name"], entity_type=ent["type"], first_seen_doc_id=doc_id)
                entity_id_map[ent["name"]] = eid
                await metadata_store.add_mention(entity_id=eid, doc_id=doc_id, chunk_index=chunk.index, context_snippet=chunk.text[:200])
            for rel in extraction.relationships:
                if not isinstance(rel, dict) or "source" not in rel or "target" not in rel:
                    continue
                source_id = entity_id_map.get(rel["source"])
                if source_id is None:
                    continue
                target_id = entity_id_map.get(rel["target"])
                if target_id is None:
                    target_id = await metadata_store.add_entity(name=rel["target"], entity_type="unknown", first_seen_doc_id=doc_id)
                    await metadata_store.add_mention(entity_id=target_id, doc_id=doc_id, chunk_index=chunk.index, context_snippet=chunk.text[:200])
                await metadata_store.add_relationship(
                    source_entity_id=source_id, target_entity_id=target_id,
                    relationship_type=rel.get("type", "related_to"), doc_id=doc_id,
                    context_snippet=chunk.text[:100],
                )
            for section in extraction.sections:
                if isinstance(section, dict) and "name" in section:
                    section_id = await metadata_store.add_entity(name=section["name"], entity_type="document_section", first_seen_doc_id=doc_id)
                    await metadata_store.add_mention(entity_id=section_id, doc_id=doc_id, chunk_index=chunk.index, context_snippet=chunk.text[:200])

        # Done
        self.complete_job(job.job_id, doc_id=doc_id, chunk_count=len(chunks))

        # Cleanup temp file
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass


# Singleton
ingest_queue = IngestQueue()
