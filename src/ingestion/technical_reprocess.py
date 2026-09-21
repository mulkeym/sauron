"""Rebuild derived technical passages from hash-verified originals, without new IDs."""
from __future__ import annotations
import asyncio
import hashlib
import tempfile
from pathlib import Path

from src.ingestion.isolation import extract_in_worker
from src.ingestion.technical_chunks import build_technical_chunks, chunk_metadata, index_text
from src.ingestion.embedder import embed_texts
from src.retrieval.models import ChunkMetadata


async def reprocess_source(doc_id, source, metadata_store, vector_store):
    from src.figures.storage import FigureStore
    from src.retrieval.vector_store import VectorStore
    import lancedb
    doc=await metadata_store.get_document(doc_id)
    if doc is None or not doc.content_hash:
        raise ValueError('An existing document with its original content hash is required')
    source=Path(source)
    def digest():
        with source.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    if await asyncio.to_thread(digest) != doc.content_hash:
        raise ValueError('Original bytes do not match; ingest changed content as a new edition')
    prepared=await extract_in_worker(source,doc.filename)
    try:
        if prepared.parsed.doc_type not in ('pdf','docx','vsdx','markdown','text'):
            raise ValueError('Technical reprocessing supports PDF, Word, Visio and text/Markdown')
        with tempfile.TemporaryDirectory(prefix='sauron-technical-index-') as temporary:
            staged=VectorStore.__new__(VectorStore)
            staged.db=lancedb.connect(temporary);staged.table_name='technical_staging';staged._table=None
            medium_count=0
            for tier,size in [('small',1024),('medium',2048),('large',4096),('xlarge',8192)]:
                chunks=build_technical_chunks(prepared,size)
                if tier=='medium':medium_count=len(chunks)
                for start in range(0,len(chunks),16):
                    batch=chunks[start:start+16]
                    texts=[index_text('',c) for c in batch]
                    metadata=[ChunkMetadata(doc_id=doc_id,filename=doc.filename,doc_type=doc.doc_type,
                        chunk_index=c.index,start_char=c.start_char,acl_groups=list(doc.acl_groups),
                        category=doc.category,chunk_size_tier=tier,**chunk_metadata(c)) for c in batch]
                    vectors=await asyncio.to_thread(embed_texts,texts)
                    await asyncio.to_thread(staged.upsert,texts,vectors,metadata)
            # Copy no figure/table-row/summary rows: their established provenance and
            # authenticated assets remain unchanged by this prose-only reprocessing.
            current=await metadata_store.get_document(doc_id)
            if (current is None or current.content_hash!=doc.content_hash
                    or current.acl_groups!=doc.acl_groups or current.dataset_id!=doc.dataset_id):
                raise ValueError('Document changed during reprocessing; retry from the current catalog')
            from src.ingestion.document_identity import analyze_document
            identity=prepared.parsed.metadata.get('document_identity') or analyze_document(prepared.parsed)
            def publish():
                rows=staged.table.to_arrow()
                if not len(rows):raise ValueError('No technical passages extracted; existing index retained')
                target=vector_store.table
                fields=target.schema.names
                rows=rows.select(fields)
                did=doc_id.replace("'","''")
                condition=f"target.doc_id = '{did}' AND target.chunk_size_tier IN ('small','medium','large','xlarge') AND target.content_type = 'text'"
                # One LanceDB transaction replaces only this document's prose rows.
                target.merge_insert('id').when_not_matched_insert_all().when_not_matched_by_source_delete(condition).execute(rows)
            await asyncio.to_thread(publish)
            figures=await metadata_store.list_figures([doc_id])
            tags={**(current.metadata_tags or {}),'document_identity':identity,
                  'technical_processing':{'schema_version':1,'structure':'section-aware'},
                  'ingestion_warnings':prepared.warnings}
            await metadata_store.update_document(doc_id,metadata_tags=tags,chunk_count=medium_count+len(figures))
        return {'doc_id':doc_id,'filename':doc.filename,'source_revision':doc.content_hash,
                'medium_passages':medium_count,'warnings':prepared.warnings}
    finally:
        FigureStore().discard(prepared.figure_staging)


async def reprocess_directory(source_dir, metadata_store, vector_store, *, dry_run=True):
    """Match accessible originals by bytes, never by a guessed filename/revision."""
    root=Path(source_dir).resolve()
    if not root.is_dir():raise ValueError('Source directory does not exist')
    docs=await metadata_store.list_documents()
    by_hash={}
    for doc in docs:
        if doc.content_hash:by_hash.setdefault(doc.content_hash,[]).append(doc)
    results=[];seen=set()
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.pdf','.docx','.vsdx','.md','.txt'):
            continue
        if not path.resolve().is_relative_to(root):continue
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        for doc in by_hash.get(digest,[]):
            if doc.doc_id in seen:continue
            seen.add(doc.doc_id)
            if dry_run:results.append({'doc_id':doc.doc_id,'filename':doc.filename,'status':'matched'})
            else:
                try:results.append(await reprocess_source(doc.doc_id,path,metadata_store,vector_store))
                except Exception as exc:results.append({'doc_id':doc.doc_id,'filename':doc.filename,'error':str(exc)})
    return {'dry_run':dry_run,'matched':len(seen),'unmatched':len(docs)-len(seen),'results':results}
