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
