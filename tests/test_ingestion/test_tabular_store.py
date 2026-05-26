"""Tests for src/ingestion/tabular_store.py — DuckDB storage + schema for clean sheets."""
import pytest

from src.ingestion.tabular_store import duckdb_table_name, _safe_column_names


def test_table_name_is_sql_safe_and_deterministic():
    n1 = duckdb_table_name("a1b2-c3d4", "Pay Rates 2024")
    n2 = duckdb_table_name("a1b2-c3d4", "Pay Rates 2024")
    assert n1 == n2                      # deterministic
    assert n1 == "doc_a1b2_c3d4_pay_rates_2024"
    assert not n1[0].isdigit()           # never starts with a digit


def test_safe_column_names_sanitizes_and_dedupes():
    cols = _safe_column_names(["GS Grade", "Step (1)", "", "Step (1)", "2024"])
    # spaces/punct -> underscores, blanks -> col_N, collisions -> suffixed, digit-leading -> prefixed
    assert cols == ["gs_grade", "step_1", "col_2", "step_1_1", "c_2024"]
