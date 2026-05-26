"""Wire clean spreadsheet sheets into the structured store + row narratives.

Called from the ingestion pipeline for spreadsheet documents. For each CLEAN
sheet: load rows into DuckDB, profile the columns (one LLM call), register +
persist the schema, and embed deterministic per-row narratives (the raw rows
stay authoritative in DuckDB). Fail-open per sheet — one bad sheet never aborts
the others or the surrounding document ingestion.
"""
import logging

from src.ingestion.tabular import read_sheets, classify_sheet
from src.ingestion.tabular_store import (
    connect_tabular, load_sheet_to_duckdb, schema_from_sheet,
)
from src.ingestion.table_profiler import profile_table, build_row_narratives
from src.ingestion.embedder import embed_texts
from src.retrieval.models import ChunkMetadata

logger = logging.getLogger(__name__)


async def ingest_spreadsheet_tables(file_path, doc_id, filename, doc_type, acl_groups,
                                    category, vector_store, metadata_store,
                                    schema_registry=None, generate_fn=None) -> int:
    """Process every CLEAN sheet of a spreadsheet. Returns the count processed."""
    if schema_registry is None:
        from src.api.routes_ingest import get_schema_registry
        schema_registry = get_schema_registry()

    try:
        grids = read_sheets(file_path)
    except Exception as e:
        logger.warning(f"Tabular ingest: could not read sheets from {filename}: {e}")
        return 0

    clean_count = 0
    chunk_index = 0
    con = None
    try:
        con = connect_tabular(read_only=False)
        for grid in grids:
            cls = classify_sheet(grid)
            if cls.route != "clean":
                continue
            try:
                _, col_names = load_sheet_to_duckdb(con, doc_id, grid.sheet_name, cls, grid)
                data_rows = grid.rows[cls.header_row_index + 1:]
                profile = profile_table(
                    grid.sheet_name, col_names, cls.column_dtypes, data_rows[:5],
                    generate_fn=generate_fn,
                )
                # Build + enrich the schema with the profile's labels/description.
                schema = schema_from_sheet(doc_id, grid.sheet_name, cls, grid, acl_groups=acl_groups)
                for col in schema.columns:
                    if col.name in profile.column_descriptions:
                        col.description = profile.column_descriptions[col.name]
                if profile.table_description:
                    schema.description = profile.table_description
                schema_registry.register(schema)
                await metadata_store.save_schema(schema)

                # Deterministic row narratives -> embeddings (raw rows live in DuckDB).
                narratives = [
                    n for n in build_row_narratives(
                        col_names, profile, data_rows, context=profile.table_description
                    ) if n.strip()
                ]
                if narratives:
                    vectors = embed_texts(narratives)
                    metadatas = []
                    for _ in narratives:
                        metadatas.append(ChunkMetadata(
                            doc_id=doc_id, filename=filename, doc_type=doc_type,
                            chunk_index=chunk_index, start_char=0, acl_groups=acl_groups,
                            category=category, chunk_size_tier="table_row",
                        ))
                        chunk_index += 1
                    vector_store.upsert(texts=narratives, vectors=vectors, metadatas=metadatas)
                clean_count += 1
            except Exception as e:
                logger.warning(f"Tabular ingest: failed on sheet '{grid.sheet_name}' of {filename}: {e}")
                continue
    finally:
        if con is not None:
            con.close()

    logger.info(f"Tabular ingest: processed {clean_count} clean sheet(s) from {filename}")
    return clean_count


async def populate_schema_registry(metadata_store, schema_registry) -> int:
    """Load every persisted TableSchema into the in-memory registry. Returns count."""
    schemas = await metadata_store.load_all_schemas()
    for schema in schemas:
        schema_registry.register(schema)
    return len(schemas)
