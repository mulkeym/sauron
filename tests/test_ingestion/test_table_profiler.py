"""Tests for src/ingestion/table_profiler.py — per-table profile + row narratives."""
from src.ingestion.table_profiler import TableProfile, _heuristic_profile


def test_heuristic_profile_splits_keys_and_measures_by_dtype():
    p = _heuristic_profile(
        ["grade", "step", "salary"],
        ["text", "number", "number"],
    )
    assert isinstance(p, TableProfile)
    # text columns are keys; number columns are measures
    assert p.key_columns == ["grade"]
    assert p.measure_columns == ["step", "salary"]
    # descriptions default to the column name itself
    assert p.column_descriptions == {"grade": "grade", "step": "step", "salary": "salary"}
    assert p.table_description  # non-empty


def test_heuristic_profile_all_text_has_no_measures():
    p = _heuristic_profile(["name", "city"], ["text", "text"])
    assert p.key_columns == ["name", "city"]
    assert p.measure_columns == []
