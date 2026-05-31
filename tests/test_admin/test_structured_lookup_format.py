"""Tests for the playground Structured Lookup step formatter."""
from src.admin.routes import _format_structured_lookup


def test_format_ran_includes_sql_gate_and_sample():
    trace = {
        "query_type": "sweep", "gate": [["all_gs", 0.71, True], ["leo", 0.18, False]],
        "sql": "SELECT * FROM all_gs WHERE locname = 'TU'", "status": "ran",
        "skip_reason": "", "error": "", "row_count": 15,
        "sample_rows": [{"grade": "GS-12", "salary": 91162}], "fell_back": False,
    }
    html = _format_structured_lookup(trace)
    assert "sweep" in html
    assert "all_gs" in html and "0.71" in html
    assert "SELECT * FROM all_gs" in html
    assert "15 rows" in html
    assert "view sample" in html


def test_format_skipped_shows_reason_no_sql():
    trace = {"query_type": "sweep", "gate": [["t", 0.1, False]], "sql": "",
             "status": "skipped", "skip_reason": "no table >= 0.3 relevance",
             "error": "", "row_count": 0, "sample_rows": [], "fell_back": False}
    html = _format_structured_lookup(trace)
    assert "skipped" in html and "no table" in html
    assert "SELECT" not in html


def test_format_skipped_notes_fallback_when_fell_back():
    """A skip that fell back should reassure the reader the question was still
    answered via document search."""
    trace = {"query_type": "analytical", "gate": None, "sql": "",
             "status": "skipped", "skip_reason": "No available table contained relevant data.",
             "error": "", "row_count": 0, "sample_rows": [], "fell_back": True}
    html = _format_structured_lookup(trace)
    assert "skipped" in html
    assert "document search" in html


def test_format_error_shows_message_and_fallback():
    trace = {"query_type": "analytical", "gate": None, "sql": "SELECT bad",
             "status": "error", "skip_reason": "", "error": "Parser Error",
             "row_count": 0, "sample_rows": [], "fell_back": True}
    html = _format_structured_lookup(trace)
    assert "error" in html and "Parser Error" in html
    assert "map-reduce" in html
    assert "no gate" in html


def test_format_zero_rows():
    trace = {"query_type": "analytical", "gate": None, "sql": "SELECT 1 WHERE 1=0",
             "status": "ran", "skip_reason": "", "error": "", "row_count": 0,
             "sample_rows": [], "fell_back": False}
    html = _format_structured_lookup(trace)
    assert "0 rows" in html
    assert "view sample" not in html


def test_format_empty_trace():
    assert "No structured lookup" in _format_structured_lookup({})
