"""Tests for the multi-turn table router.

The router bounds the text-to-SQL schema prompt by selecting only the tables
relevant to a question BEFORE the (expensive, value-dumping) schema prompt is
rendered — so the prompt stays within the model context as the corpus grows to
hundreds/thousands of tables.
"""
import json

import pytest

from src.agent.strategies import structured
from src.agent.strategies.structured import select_relevant_tables
from src.db.schema_registry import TableSchema, ColumnSchema


def _schema(table, desc="", cols=("a",)):
    return TableSchema(database="db", table=table,
                       columns=[ColumnSchema(c, "VARCHAR", "") for c in cols],
                       description=desc, acl_groups=["ALL"])


def _wide_budget(monkeypatch):
    """Set a huge catalog budget so happy-path routing never triggers the
    embedding pre-rank (real embeddings are unavailable in CI)."""
    from src.config import settings
    monkeypatch.setattr(settings, "sql_table_routing_enabled", True)
    monkeypatch.setattr(settings, "sql_table_routing_catalog_budget_chars", 10_000_000)


def test_small_corpus_passes_through_without_llm(monkeypatch):
    """No more tables than the per-prompt cap -> return all, never call the LLM."""
    from src.config import settings
    monkeypatch.setattr(settings, "sql_table_routing_max_selected", 8)

    def _boom(**kw):
        raise AssertionError("router LLM must not be called for a small corpus")

    schemas = [_schema(f"t{i}") for i in range(3)]
    assert select_relevant_tables("q", schemas, generate_fn=_boom) == schemas


def test_disabled_returns_all_without_llm(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "sql_table_routing_enabled", False)
    monkeypatch.setattr(settings, "sql_table_routing_max_selected", 2)

    def _boom(**kw):
        raise AssertionError("router LLM must not be called when disabled")

    schemas = [_schema(f"t{i}") for i in range(50)]
    assert select_relevant_tables("q", schemas, generate_fn=_boom) == schemas


def test_routes_large_corpus_to_named_tables(monkeypatch):
    from src.config import settings
    _wide_budget(monkeypatch)
    monkeypatch.setattr(settings, "sql_table_routing_max_selected", 8)

    schemas = [_schema(f"t{i}", desc=f"desc{i}", cols=("x", "y")) for i in range(50)]
    captured = {}

    def _gen(**kw):
        captured["user"] = kw["user_prompt"]
        return '["t7", "t42"]'

    out = select_relevant_tables("find t7", schemas, generate_fn=_gen)
    assert [s.table for s in out] == ["t7", "t42"]
    # The routing catalog is COMPACT: table + column names, no value dumps.
    assert "t7" in captured["user"]
    assert "columns: x, y" in captured["user"]


def test_caps_selection_at_max_selected(monkeypatch):
    from src.config import settings
    _wide_budget(monkeypatch)
    monkeypatch.setattr(settings, "sql_table_routing_max_selected", 3)

    schemas = [_schema(f"t{i}") for i in range(50)]
    names = [f"t{i}" for i in range(10)]
    out = select_relevant_tables("q", schemas, generate_fn=lambda **kw: json.dumps(names))
    assert [s.table for s in out] == ["t0", "t1", "t2"]


def test_ignores_hallucinated_table_names(monkeypatch):
    """Names the model invents that aren't in the catalog are dropped."""
    from src.config import settings
    _wide_budget(monkeypatch)
    monkeypatch.setattr(settings, "sql_table_routing_max_selected", 8)

    schemas = [_schema(f"t{i}") for i in range(50)]
    out = select_relevant_tables("q", schemas, generate_fn=lambda **kw: '["t9", "nonexistent"]')
    assert [s.table for s in out] == ["t9"]


def test_fail_open_to_bounded_topk_on_garbled_output(monkeypatch):
    """A non-JSON / unusable router response must NOT send the whole corpus to
    the SQL prompt — fall back to a bounded embedding top-K."""
    from src.config import settings
    _wide_budget(monkeypatch)
    monkeypatch.setattr(settings, "sql_table_routing_max_selected", 5)

    schemas = [_schema(f"t{i}") for i in range(50)]
    # All tables equally similar -> stable order -> top-5 == first 5.
    eq = lambda q: [1.0, 0.0]
    et = lambda texts: [[1.0, 0.0] for _ in texts]
    out = select_relevant_tables("q", schemas, generate_fn=lambda **kw: "I cannot help",
                                 embed_query_fn=eq, embed_texts_fn=et)
    assert [s.table for s in out] == ["t0", "t1", "t2", "t3", "t4"]


def test_router_can_decline_with_empty_array(monkeypatch):
    """An EXPLICIT empty array means the model judged NO table relevant -> return
    [] (the structured path will skip). This must NOT be treated as a garbled
    response that fails open to a top-K (which would force irrelevant tables)."""
    from src.config import settings
    _wide_budget(monkeypatch)
    monkeypatch.setattr(settings, "sql_table_routing_max_selected", 8)

    schemas = [_schema(f"t{i}") for i in range(50)]
    # Embeddings WOULD yield tables if we fell open — prove the decline is honored.
    eq = lambda q: [1.0, 0.0]
    et = lambda texts: [[1.0, 0.0] for _ in texts]
    out = select_relevant_tables("unanswerable", schemas, generate_fn=lambda **kw: "[]",
                                 embed_query_fn=eq, embed_texts_fn=et)
    assert out == []


def test_empty_schemas_returns_empty():
    assert select_relevant_tables("q", [], generate_fn=lambda **kw: "[]") == []
