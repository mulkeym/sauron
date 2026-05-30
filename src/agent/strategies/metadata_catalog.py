"""File-metadata (catalog) Q&A: text-to-SQL over an ACL-pre-filtered in-memory
DuckDB catalog of the documents the asking user can access."""
import logging
from collections import Counter
from datetime import date, datetime
from urllib.parse import unquote

from src.agent.strategies.structured import generate_sql, StructuredLookupTrace
from src.ingestion.tabular_store import execute_duckdb_sql
from src.retrieval.models import RetrievedChunk, ChunkMetadata

logger = logging.getLogger(__name__)

CATALOG_SCHEMA = """Table "files" — one row per document the user can access:
  doc_id VARCHAR, filename VARCHAR, doc_type VARCHAR (e.g. 'pdf','xlsx','docx'),
  dataset VARCHAR (dataset name), category VARCHAR, uploaded_by VARCHAR,
  created_at TIMESTAMPTZ (when the file was uploaded), chunk_count INTEGER,
  summary VARCHAR, source_url VARCHAR (origin URL for web-connector files; '' if uploaded directly),
  tags VARCHAR (lowercased extracted entities/organizations/amounts/topics/identifiers).
Table "datasets" — name VARCHAR, description VARCHAR.
Table "categories" — name VARCHAR, description VARCHAR.
Rules: use ILIKE for case-insensitive text/tag matching; use COUNT for "how many";
filenames are stored human-readable (spaces, not %20) — match them with ILIKE and
wildcards (e.g. filename ILIKE '%newsletter january 2021%'), not exact =, since the
user may give a partial or differently-spaced name;
when listing specific files, always SELECT filename AND doc_id;
add LIMIT 100 when listing rows, unless the question asks for all rows or for a count."""


def _clean_filename(name) -> str:
    """Decode URL-encoded filenames (web-connector docs are stored with %20 etc.)
    so the catalog exposes human-readable names that match natural-language queries."""
    return unquote(name or "")


def _flatten_tags(metadata_tags) -> str:
    """Flatten the extracted metadata_tags dict into one lowercased searchable string."""
    if not metadata_tags:
        return ""
    vals = []
    for field in ("entities", "organizations", "amounts", "topics", "identifiers"):
        for v in (metadata_tags.get(field) or []):
            vals.append(str(v))
    return " ".join(vals).lower()


def build_catalog_connection(docs, dataset_names=None, datasets=None, categories=None):
    """Build an ephemeral in-memory DuckDB with a `files` table (one row per doc)
    and optional `datasets`/`categories` tables. enable_external_access=False so
    LLM-generated SELECTs cannot reach the filesystem or network."""
    import duckdb
    dataset_names = dataset_names or {}
    con = duckdb.connect(":memory:", config={"enable_external_access": False})
    con.execute("""CREATE TABLE files (
        doc_id VARCHAR, filename VARCHAR, doc_type VARCHAR, dataset VARCHAR,
        category VARCHAR, uploaded_by VARCHAR, created_at TIMESTAMPTZ,
        chunk_count INTEGER, summary VARCHAR, source_url VARCHAR, tags VARCHAR)""")
    for d in docs:
        con.execute(
            "INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [d.doc_id, _clean_filename(d.filename), d.doc_type,
             dataset_names.get(getattr(d, "dataset_id", 0), ""),
             getattr(d, "category", "") or "",
             getattr(d, "uploaded_by", "") or "",
             getattr(d, "created_at", None),
             getattr(d, "chunk_count", 0) or 0,
             getattr(d, "summary", "") or "",
             getattr(d, "source_url", "") or "",
             _flatten_tags(getattr(d, "metadata_tags", None))])
    if datasets is not None:
        con.execute("CREATE TABLE datasets (name VARCHAR, description VARCHAR)")
        for ds in datasets:
            con.execute("INSERT INTO datasets VALUES (?,?)",
                        [ds.name, getattr(ds, "description", "") or ""])
    if categories is not None:
        con.execute("CREATE TABLE categories (name VARCHAR, description VARCHAR)")
        for c in categories:
            con.execute("INSERT INTO categories VALUES (?,?)",
                        [c.name, getattr(c, "description", "") or ""])
    return con


_ALLOWED = {"files", "datasets", "categories"}


def _json_safe_rows(rows):
    """Convert datetime/date cell values to ISO strings so result rows are JSON-serializable
    (the catalog `created_at` column is a TIMESTAMPTZ)."""
    return [
        {k: (v.isoformat() if isinstance(v, (datetime, date)) else v) for k, v in r.items()}
        for r in rows
    ]


def _citations_from_rows(rows, docs) -> list:
    """Build file citations for result rows that name a real document (doc_id)."""
    by_id = {d.doc_id: d for d in docs}
    chunks, seen = [], set()
    for r in rows:
        did = r.get("doc_id")
        if did and did in by_id and did not in seen:
            seen.add(did)
            d = by_id[did]
            chunks.append(RetrievedChunk(
                text=f"{d.filename} (type: {d.doc_type}): {getattr(d, 'summary', '') or ''}",
                score=0.9,  # catalog hits rank high vs. vector chunks
                metadata=ChunkMetadata(
                    doc_id=d.doc_id, filename=d.filename, doc_type=d.doc_type,
                    chunk_index=0, start_char=0,
                    acl_groups=getattr(d, "acl_groups", []) or []),
            ))
    return chunks


def _catalog_snapshot_chunk(docs) -> RetrievedChunk:
    """ACL-filtered prose snapshot of the catalog: totals + rollups + a file list."""
    by_type = Counter(d.doc_type for d in docs)
    lines = [f"Total documents you can access: {len(docs)}",
             "By type: " + ", ".join(f"{k}: {v}" for k, v in sorted(by_type.items()))]
    for d in docs[:50]:  # cap the inline list to keep the snapshot within the LLM context budget
        lines.append(f"- {d.filename} (type {d.doc_type}, {getattr(d, 'chunk_count', 0) or 0} chunks)")
    return RetrievedChunk(
        text="Document catalog (metadata):\n" + "\n".join(lines),
        score=0.3,
        metadata=ChunkMetadata(
            doc_id="metadata-context", filename="metadata_context",
            doc_type="metadata", chunk_index=0, start_char=0, acl_groups=["ALL"]),  # safe: the doc list is already ACL-filtered upstream by list_documents(user_groups)
    )


async def retrieve_metadata_catalog(state, metadata_store=None, generate_fn=None) -> dict:
    """Answer a catalog question via text-to-SQL over the ACL-filtered in-memory
    catalog. Falls back to a prose snapshot on SQL error or empty result."""
    question = state["question"]
    user_groups = state.get("user_groups", [])
    if metadata_store is None:
        from src.api.routes_ingest import get_metadata_store
        metadata_store = get_metadata_store()

    docs = await metadata_store.list_documents(user_groups)   # ACL boundary
    try:
        datasets = await metadata_store.list_datasets(active_only=True)
    except Exception:
        datasets = []
    try:
        categories = await metadata_store.list_categories()
    except Exception:
        categories = []
    dataset_names = {getattr(ds, "id", 0): ds.name for ds in datasets}

    trace = StructuredLookupTrace(query_type="metadata")
    rows = []
    try:
        con = build_catalog_connection(docs, dataset_names, datasets, categories)
        try:
            trace.sql = generate_sql(CATALOG_SCHEMA, question, generate_fn=generate_fn)
            rows = execute_duckdb_sql(con, trace.sql, allowed_tables=_ALLOWED)
            rows = _json_safe_rows(rows)
            trace.status = "ran"
            trace.rows = rows
            trace.row_count = len(rows)
            trace.sample_rows = rows[:5]
        finally:
            con.close()
    except Exception as e:
        trace.status = "error"
        trace.fell_back = True
        trace.error = str(e)
        logger.warning("Metadata catalog SQL failed: %s", e)

    attempts = state.get("retrieval_attempts", 0) + 1
    if trace.status == "ran" and rows:
        return {"retrieved_chunks": _citations_from_rows(rows, docs),
                "sql_results": rows, "structured_trace": trace.to_dict(),
                "retrieval_attempts": attempts}

    # Fallback: prose snapshot so we answer "what I can see" instead of "no data".
    trace.fell_back = True
    return {"retrieved_chunks": [_catalog_snapshot_chunk(docs)],
            "sql_results": [], "structured_trace": trace.to_dict(),
            "retrieval_attempts": attempts}
