"""Add images to an existing source version without deleting its text or ACLs."""
import asyncio
import hashlib
from src.figures.storage import FigureStore, track_ingestion

_lock = asyncio.Lock()


@track_ingestion
async def backfill_figures(doc_id, source, metadata_store, vector_store):
    from src.ingestion.queue import ingest_queue
    from src.ingestion.isolation import extract_in_worker
    from src.ingestion.embedder import embed_texts
    from src.retrieval.models import ChunkMetadata
    if _lock.locked() or ingest_queue.has_active_jobs():
        raise ValueError("Wait for active ingestion or figure backfill to complete.")
    async with _lock:
        doc = await metadata_store.get_document(doc_id)
        if doc is None:
            raise ValueError("Document not found")
        if not doc.content_hash:
            raise ValueError("This document has no original content hash. Re-ingest from the managed source to establish its version.")
        def digest():
            with source.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()
        if await asyncio.to_thread(digest) != doc.content_hash:
            raise ValueError("Source content differs from the ingested document; use source replacement instead.")
        if await metadata_store.list_figures([doc_id]):
            raise ValueError("This document already has stored figures. Backfill only adds images to documents without stored figures.")
        prepared = await extract_in_worker(source, doc.filename)
        store = FigureStore()
        published = False
        inserted = []
        try:
            previous = await asyncio.to_thread(vector_store.figure_row_ids, doc_id)
            records = prepared.pdf.figure_records if prepared.pdf else (prepared.office.figures if prepared.office else [])
            records = [r for r in records if r.assets]
            # New extractor versions can change positional figure IDs. Never
            # associate old inline descriptions/cached citations with new pixels.
            for record in records:
                record.figure_id = "stored-" + record.figure_id
            if not records:
                return {"doc_id": doc_id, "stored": 0, "warnings": prepared.warnings + ["No retainable figures were detected."]}
            texts = [f"Document: {doc.filename}\n\n{r.retrieval_text()}" for r in records]
            vectors = await asyncio.to_thread(embed_texts, texts)
            metas = [ChunkMetadata(doc_id=doc_id, filename=doc.filename, doc_type=doc.doc_type,
                chunk_index=doc.chunk_count + i, start_char=0, acl_groups=list(doc.acl_groups),
                category=doc.category, content_type="figure", figure_id=r.figure_id, figure_kind=r.kind,
                page=r.page + 1 if r.page is not None else None, slide=r.slide + 1 if r.slide is not None else None,
                caption=r.caption, source_locator=f"Figure {r.figure_id}") for i, r in enumerate(records)]
            # Add new entries before retiring older text-only figure entries.
            # If indexing fails, no new asset metadata is published or accessible.
            indexing = asyncio.create_task(asyncio.to_thread(vector_store.upsert, texts, vectors, metas))
            try:
                inserted = await asyncio.shield(indexing)
            except asyncio.CancelledError:
                inserted = await indexing
                raise
            figures = await store.publish_async(doc_id, records, prepared.figure_staging)
            current = await metadata_store.get_document(doc_id)
            if current is None or current.content_hash != doc.content_hash:
                raise ValueError("Document changed during backfill")
            commit = asyncio.create_task(metadata_store.put_figures(doc_id, figures))
            try:
                await asyncio.shield(commit)
            except asyncio.CancelledError:
                await commit
                published = True
                raise
            published = True
            try:
                await asyncio.to_thread(vector_store.delete_ids, previous)
            except Exception:
                prepared.warnings.append("Older figure index rows could not be retired; search results remain deduplicated.")
            return {"doc_id": doc_id, "stored": len(figures), "warnings": prepared.warnings}
        finally:
            store.discard(prepared.figure_staging)
            if not published:
                await asyncio.to_thread(vector_store.delete_ids, inserted)
                store.delete_document(doc_id)
