"""Tests for src/ingestion/table_profiler.py — per-table profile + row narratives."""
import json

from src.ingestion.table_profiler import TableProfile, _heuristic_profile, profile_table, _fmt_cell, row_narrative, build_row_narratives
from src.ingestion.table_profiler import glossary_lookup


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


_COLS = ["grade", "step", "salary"]
_DTYPES = ["text", "number", "number"]
_SAMPLE = [["GS-12", 5, 86415], ["GS-13", 5, 102000]]


def _fake_generate(payload):
    """Return a generate_fn that always yields json.dumps(payload)."""
    def _gen(system_prompt, user_prompt, **kwargs):
        return json.dumps(payload)
    return _gen


def test_profile_table_uses_llm_output():
    gen = _fake_generate({
        "column_descriptions": {"grade": "Pay grade", "step": "Step", "salary": "Annual salary"},
        "key_columns": ["grade", "step"],
        "measure_columns": ["salary"],
        "table_description": "GS pay by grade and step",
    })
    p = profile_table("Pay", _COLS, _DTYPES, _SAMPLE, generate_fn=gen)
    assert p.key_columns == ["grade", "step"]
    assert p.measure_columns == ["salary"]
    assert p.column_descriptions["salary"] == "Annual salary"
    assert p.table_description == "GS pay by grade and step"


def test_profile_table_drops_columns_not_in_table():
    gen = _fake_generate({
        "column_descriptions": {"grade": "Pay grade"},
        "key_columns": ["grade", "bogus"],     # bogus is not a real column
        "measure_columns": ["salary", "also_fake"],
        "table_description": "x",
    })
    p = profile_table("Pay", _COLS, _DTYPES, _SAMPLE, generate_fn=gen)
    assert p.key_columns == ["grade"]          # bogus filtered out
    assert p.measure_columns == ["salary"]     # also_fake filtered out
    # columns the LLM didn't describe still get a default label
    assert p.column_descriptions["step"] == "step"


def test_profile_table_falls_back_when_llm_raises():
    def boom(system_prompt, user_prompt, **kwargs):
        raise RuntimeError("LLM down")
    p = profile_table("Pay", _COLS, _DTYPES, _SAMPLE, generate_fn=boom)
    # heuristic fallback: number cols -> measures
    assert p.key_columns == ["grade"]
    assert p.measure_columns == ["step", "salary"]


def test_profile_table_falls_back_on_unparseable_output():
    def junk(system_prompt, user_prompt, **kwargs):
        return "not json at all"
    p = profile_table("Pay", _COLS, _DTYPES, _SAMPLE, generate_fn=junk)
    assert p.key_columns == ["grade"]
    assert p.measure_columns == ["step", "salary"]


def test_profile_table_falls_back_on_non_dict_json():
    # Valid JSON but a list, not an object -> data.get(...) fails -> heuristic.
    gen = _fake_generate([{"key": "val"}])
    p = profile_table("Pay", _COLS, _DTYPES, _SAMPLE, generate_fn=gen)
    assert p.key_columns == ["grade"]
    assert p.measure_columns == ["step", "salary"]


_PROFILE = TableProfile(
    column_descriptions={"grade": "Pay grade", "step": "Step", "salary": "Annual salary"},
    key_columns=["grade", "step"],
    measure_columns=["salary"],
    table_description="GS pay",
)


def test_fmt_cell():
    assert _fmt_cell(None) == "(not specified)"
    assert _fmt_cell("") == "(not specified)"
    assert _fmt_cell("  ") == "(not specified)"
    assert _fmt_cell("GS-12") == "GS-12"
    assert _fmt_cell(5) == "5"
    assert _fmt_cell(0) == "0"
    assert _fmt_cell(0.0) == "0.0"
    assert _fmt_cell(False) == "False"


def test_row_narrative_uses_labels_and_keys_then_measures():
    text = row_narrative(["grade", "step", "salary"], _PROFILE, ["GS-12", 5, 86415])
    assert text == "Pay grade=GS-12, Step=5: Annual salary is 86415"


def test_row_narrative_marks_missing_cells():
    # row shorter than the columns -> missing cells are explicit, never fabricated
    text = row_narrative(["grade", "step", "salary"], _PROFILE, ["GS-12"])
    assert text == "Pay grade=GS-12, Step=(not specified): Annual salary is (not specified)"


def test_row_narrative_measures_only():
    profile = TableProfile(column_descriptions={"x": "X"}, key_columns=[], measure_columns=["x"])
    assert row_narrative(["x"], profile, [42]) == "X is 42"


def test_build_row_narratives_one_per_row_with_context():
    rows = [["GS-12", 5, 86415], ["GS-13", 5, 102000]]
    out = build_row_narratives(["grade", "step", "salary"], _PROFILE, rows, context="GS pay")
    assert out == [
        "GS pay — Pay grade=GS-12, Step=5: Annual salary is 86415",
        "GS pay — Pay grade=GS-13, Step=5: Annual salary is 102000",
    ]


def test_build_row_narratives_without_context():
    out = build_row_narratives(["grade", "step", "salary"], _PROFILE, [["GS-12", 5, 86415]])
    assert out == ["Pay grade=GS-12, Step=5: Annual salary is 86415"]


def test_build_row_narratives_empty_rows():
    assert build_row_narratives(["grade"], _PROFILE, []) == []


def test_glossary_lookup_exact_then_prefix():
    g = {"GS": "base", "E-*": "Enlisted Member", "O-*": "Commissioned Officer"}
    assert glossary_lookup(g, "GS") == "base"          # exact wins
    assert glossary_lookup(g, "E-3") == "Enlisted Member"
    assert glossary_lookup(g, "O-10") == "Commissioned Officer"
    assert glossary_lookup(g, "W-2") is None           # no match
