"""Tests for the orphan-purge decision logic and its safety guards."""
from src.admin.orphans import plan_orphan_purge


def test_refuses_when_ingestion_active():
    # Even with a clear orphan, an active ingestion blocks the purge.
    status, orphans = plan_orphan_purge(
        metadata_doc_ids={"a"}, chunk_doc_ids={"a", "b"}, ingestion_active=True
    )
    assert status == "refused_active"
    assert orphans == set()


def test_refuses_when_metadata_empty():
    # The footgun: empty metadata must NOT mean "delete everything".
    status, orphans = plan_orphan_purge(
        metadata_doc_ids=set(), chunk_doc_ids={"a", "b", "c"}, ingestion_active=False
    )
    assert status == "refused_empty_metadata"
    assert orphans == set()


def test_identifies_only_unowned_chunks():
    status, orphans = plan_orphan_purge(
        metadata_doc_ids={"a", "b"},
        chunk_doc_ids={"a", "b", "orphan1", "orphan2"},
        ingestion_active=False,
    )
    assert status == "ok"
    assert orphans == {"orphan1", "orphan2"}


def test_ok_with_no_orphans():
    status, orphans = plan_orphan_purge(
        metadata_doc_ids={"a", "b"}, chunk_doc_ids={"a", "b"}, ingestion_active=False
    )
    assert status == "ok"
    assert orphans == set()
