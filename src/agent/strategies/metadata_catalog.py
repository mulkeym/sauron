"""File-metadata (catalog) Q&A: text-to-SQL over an ACL-pre-filtered in-memory
DuckDB catalog of the documents the asking user can access."""
import logging

logger = logging.getLogger(__name__)

CATALOG_SCHEMA = """Table "files" — one row per document the user can access:
  doc_id VARCHAR, filename VARCHAR, doc_type VARCHAR (e.g. 'pdf','xlsx','docx'),
  dataset VARCHAR (dataset name), category VARCHAR, uploaded_by VARCHAR,
  created_at TIMESTAMPTZ (when the file was uploaded), chunk_count INTEGER,
  summary VARCHAR, tags VARCHAR (lowercased extracted entities/organizations/amounts/topics/identifiers).
Table "datasets" — name VARCHAR, description VARCHAR.
Table "categories" — name VARCHAR, description VARCHAR.
Rules: use ILIKE for case-insensitive text/tag matching; use COUNT for "how many";
when listing specific files, always SELECT filename AND doc_id."""


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
        chunk_count INTEGER, summary VARCHAR, tags VARCHAR)""")
    for d in docs:
        con.execute(
            "INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,?)",
            [d.doc_id, d.filename, d.doc_type,
             dataset_names.get(getattr(d, "dataset_id", 0), ""),
             getattr(d, "category", "") or "",
             getattr(d, "uploaded_by", "") or "",
             getattr(d, "created_at", None),
             getattr(d, "chunk_count", 0) or 0,
             getattr(d, "summary", "") or "",
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
