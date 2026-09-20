"""Exact passage reads for MCP clients; no browser sessions or public chunk URLs."""
import asyncio
import re

from src.agent.state import chunk_key
from src.agent.synthesizer import _evidence_id
from src.sources.service import OriginalError, authorized_document


async def cited_passage(doc_id, revision, evidence_id, chunk_index, chunk_size_tier,
                        start_char, groups, metadata_store, vector_store):
    try:
        doc = await authorized_document(doc_id, revision, groups, metadata_store)
        if (not re.fullmatch(r'E[0-9a-f]{12}', evidence_id) or chunk_index < 0
                or start_char < 0 or not re.fullmatch(r'[a-z_]{1,32}', chunk_size_tier)):
            raise OriginalError(404, 'Cited passage not found')
        chunks = await asyncio.to_thread(
            vector_store.read_citation_candidates, doc_id, groups,
            chunk_index=chunk_index, chunk_size_tier=chunk_size_tier, start_char=start_char,
        )
        for chunk in chunks:
            if _evidence_id(repr(chunk_key(chunk)), chunk.text) == evidence_id:
                # Recheck if catalog permissions/revision changed during the index read.
                await authorized_document(doc_id, revision, groups, metadata_store)
                return {'available': True, 'doc_id': doc_id, 'source_revision': revision,
                        'evidence_id': evidence_id, 'filename': doc.filename,
                        'snippet': chunk.text, 'chunk_index': chunk_index,
                        'chunk_size_tier': chunk_size_tier, 'start_char': start_char}
        raise OriginalError(404, 'Exact cited passage is no longer indexed')
    except OriginalError as exc:
        return {'available': False, 'status': exc.status, 'error': exc.detail}
