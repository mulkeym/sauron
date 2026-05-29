"""Catalog-build helpers for file-metadata Q&A."""
from types import SimpleNamespace
from datetime import datetime, timezone

from src.agent.strategies.metadata_catalog import (
    _flatten_tags, build_catalog_connection, CATALOG_SCHEMA,
)
from src.ingestion.tabular_store import execute_duckdb_sql


def _doc(doc_id="d1", filename="pay.pdf", doc_type="pdf", dataset_id=2,
         category="payroll", tags=None):
    return SimpleNamespace(
        doc_id=doc_id, filename=filename, doc_type=doc_type, dataset_id=dataset_id,
        category=category, acl_groups=["executives"], chunk_count=5,
        source_url="", summary="active duty pay", uploaded_by="mike",
        created_at=datetime(2025, 5, 1, tzinfo=timezone.utc),
        metadata_tags=(tags if tags is not None else {"organizations": ["DoD"], "topics": ["Officer pay"]}),
    )


def test_flatten_tags_lowercases_and_joins():
    out = _flatten_tags({"organizations": ["DoD"], "topics": ["Officer Pay"], "amounts": ["$5,000"]})
    assert "dod" in out and "officer pay" in out and "$5,000" in out


def test_flatten_tags_empty():
    assert _flatten_tags(None) == ""
    assert _flatten_tags({}) == ""


def test_build_catalog_has_files_table_with_columns():
    con = build_catalog_connection([_doc()], dataset_names={2: "DoD OPM Policies"})
    rows = execute_duckdb_sql(con, "SELECT doc_id, filename, doc_type, dataset, category, "
                                   "uploaded_by, chunk_count, summary, tags FROM files",
                              allowed_tables={"files"})
    assert len(rows) == 1
    r = rows[0]
    assert r["filename"] == "pay.pdf"
    assert r["dataset"] == "DoD OPM Policies"
    assert "dod" in r["tags"]


def test_build_catalog_count_aggregate():
    docs = [_doc(doc_id="a", doc_type="pdf"), _doc(doc_id="b", doc_type="pdf"),
            _doc(doc_id="c", doc_type="xlsx")]
    con = build_catalog_connection(docs, dataset_names={2: "DoD"})
    rows = execute_duckdb_sql(con, "SELECT doc_type, COUNT(*) AS n FROM files GROUP BY doc_type ORDER BY doc_type",
                              allowed_tables={"files"})
    assert rows == [{"doc_type": "pdf", "n": 2}, {"doc_type": "xlsx", "n": 1}]


def test_build_catalog_tag_search_ilike():
    docs = [_doc(doc_id="a", tags={"topics": ["Officer pay"]}),
            _doc(doc_id="b", tags={"topics": ["Enlisted pay"]})]
    con = build_catalog_connection(docs, dataset_names={2: "DoD"})
    rows = execute_duckdb_sql(con, "SELECT doc_id FROM files WHERE tags ILIKE '%officer%'",
                              allowed_tables={"files"})
    assert rows == [{"doc_id": "a"}]


def test_build_catalog_optional_datasets_categories():
    con = build_catalog_connection(
        [_doc()], dataset_names={2: "DoD"},
        datasets=[SimpleNamespace(name="DoD", description="policies")],
        categories=[SimpleNamespace(name="payroll", description="pay records")])
    ds = execute_duckdb_sql(con, "SELECT name FROM datasets", allowed_tables={"datasets"})
    cat = execute_duckdb_sql(con, "SELECT name FROM categories", allowed_tables={"categories"})
    assert ds == [{"name": "DoD"}] and cat == [{"name": "payroll"}]


def test_catalog_schema_mentions_files_and_tags():
    assert "files" in CATALOG_SCHEMA and "tags" in CATALOG_SCHEMA


def test_build_catalog_created_at_date_filter():
    docs = [_doc(doc_id="may")]  # created_at = 2025-05-01 (UTC)
    apr = _doc(doc_id="apr")
    apr.created_at = datetime(2025, 4, 1, tzinfo=timezone.utc)
    con = build_catalog_connection([*docs, apr], dataset_names={2: "DoD"})
    rows = execute_duckdb_sql(
        con,
        "SELECT doc_id FROM files WHERE created_at >= '2025-05-01' AND created_at < '2025-06-01'",
        allowed_tables={"files"})
    assert rows == [{"doc_id": "may"}]
