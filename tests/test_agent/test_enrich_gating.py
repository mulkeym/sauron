"""Knowledge-graph enrichment must be skipped for METADATA (catalog) queries."""
from src.agent.graph import _skip_enrich
from src.agent.state import QueryType


def test_skip_enrich_for_metadata():
    assert _skip_enrich({"query_type": QueryType.METADATA}) is True


def test_skip_enrich_for_metadata_string():
    # query_type may be the raw string value in some states
    assert _skip_enrich({"query_type": "metadata"}) is True


def test_skip_enrich_when_skip_graph_flag():
    assert _skip_enrich({"skip_graph": True}) is True


def test_no_skip_for_normal_query():
    assert _skip_enrich({"query_type": QueryType.LOOKUP}) is False
    assert _skip_enrich({}) is False
