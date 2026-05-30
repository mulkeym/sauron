import json
from src.agent.strategies import structured as S


def test_classify_empty():
    assert S._classify_sql_result([]) == "empty"


def test_classify_degenerate_all_null():
    rows = [{"a": None, "b": None}, {"a": None, "b": None}]
    assert S._classify_sql_result(rows) == "degenerate"


def test_classify_satisfactory():
    rows = [{"locality": "Tampa", "salary": 86415}]
    assert S._classify_sql_result(rows) == "satisfactory"


def test_classify_too_large(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "sql_result_budget_chars", 50)
    monkeypatch.setattr(settings, "llm_max_context", 200000)
    rows = [{"locality": f"loc{i}", "salary": i} for i in range(100)]
    assert S._classify_sql_result(rows) == "too_large"


def test_feedback_too_large_mentions_aggregate():
    rows = [{"a": i} for i in range(885)]
    fb = S._repair_feedback("too_large", rows=rows, sql="SELECT * FROM t", question="q")
    assert "885" in fb
    assert "aggregate" in fb.lower()
    assert "SELECT * FROM t" in fb


def test_feedback_empty_mentions_filter():
    fb = S._repair_feedback("empty", rows=[], sql="SELECT * FROM t WHERE x='z'", question="q")
    assert "no rows" in fb.lower()
    assert "filter" in fb.lower()


def test_feedback_error_includes_error():
    fb = S._repair_feedback("error", rows=[], sql="SELEKT", question="q", error="syntax error")
    assert "syntax error" in fb


def test_feedback_includes_judge_reason():
    rows = [{"a": i} for i in range(885)]
    fb = S._repair_feedback("too_large", rows=rows, sql="SELECT * FROM t",
                            question="q", judge_reason="wrong column")
    assert "wrong column" in fb
