"""Index already-extracted content without opening the source document."""
import logging

from src.ingestion.tabular_chunker import sheets_needing_text
from src.ingestion.tabular_ingest import ingest_grids, SPREADSHEET_DOC_TYPES

logger = logging.getLogger(__name__)


async def index_prepared(prepared, doc_id, acl_groups, category, vector_store,
                         metadata_store, dataset_id=0):
    parsed = prepared.parsed
    for warning in prepared.warnings:
        logger.warning("%s: %s", parsed.filename, warning)
    text_sheets = None
    enriched_prose = None
    figure_records = []

    async def store_grids(grids):
        return await ingest_grids(
            grids, doc_id, parsed.filename, parsed.doc_type,
            acl_groups, category, vector_store, metadata_store, dataset_id=dataset_id,
        )

    if parsed.doc_type in SPREADSHEET_DOC_TYPES:
        grids = prepared.spreadsheet_grids
        classifications, ingested = await store_grids(grids)
        text_sheets = sheets_needing_text(grids, classifications, ingested)
        if prepared.office:
            if prepared.office.table_grids:
                await store_grids(prepared.office.table_grids)
            figure_records = prepared.office.figures
            if prepared.office.enriched_text.strip():
                enriched_prose = ((parsed.text or "") + "\n\n## Embedded figures\n\n"
                                  + prepared.office.enriched_text).strip()
    elif prepared.pdf is not None:
        try:
            await store_grids(prepared.pdf.table_grids)
            enriched_prose = "\n\n".join(b.text for b in prepared.pdf.prose_blocks)
            figure_records = prepared.pdf.figure_records
        except Exception as exc:
            logger.warning("PDF table indexing failed for %s; using flat text: %s", parsed.filename, exc)
    elif prepared.office is not None:
        enriched_prose = prepared.office.enriched_text
        figure_records = prepared.office.figures
        if prepared.office.table_grids:
            await store_grids(prepared.office.table_grids)
    return text_sheets, enriched_prose, figure_records
