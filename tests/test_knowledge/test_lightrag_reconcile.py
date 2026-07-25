"""LightRAG orphan planning + purge helpers (crash / stale doc recovery)."""
from src.knowledge.graph_rag import plan_orphan_lightrag_docs


def test_plan_orphan_drops_docs_not_in_metadata():
    live = {"doc-a", "doc-b"}
    lightrag = {"doc-a", "stale-pdf", "another-orphan"}
    assert plan_orphan_lightrag_docs(live, lightrag) == {"stale-pdf", "another-orphan"}


def test_plan_orphan_empty_metadata_drops_all():
    assert plan_orphan_lightrag_docs(set(), {"x", "y"}) == {"x", "y"}


def test_plan_orphan_keeps_exact_live_set():
    live = {"a", "b", "c"}
    assert plan_orphan_lightrag_docs(live, {"a", "b", "c"}) == set()


def test_plan_orphan_ignores_empty_ids():
    assert plan_orphan_lightrag_docs({""}, {"", "real"}) == {"real"}
