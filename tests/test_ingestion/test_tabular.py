"""Tests for src/ingestion/tabular.py — structured parse + clean/messy classification."""
from pathlib import Path

import openpyxl
import pytest

from src.ingestion.tabular import SheetGrid, read_sheets
from src.ingestion.tabular import _cell_kind
from src.ingestion.tabular import detect_header_row
from src.ingestion.tabular import infer_column_dtypes
from src.ingestion.tabular import SheetClassification, classify_sheet


def _write_xlsx(path: Path, sheets: dict[str, list[list]]):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for row in rows:
            ws.append(row)
    wb.save(path)


def test_read_sheets_xlsx_returns_one_grid_per_sheet(tmp_path):
    p = tmp_path / "book.xlsx"
    _write_xlsx(p, {
        "Pay": [["grade", "step", "salary"], ["GS-12", 5, 86415]],
        "Notes": [["just a note"]],
    })
    grids = read_sheets(p)
    assert [g.sheet_name for g in grids] == ["Pay", "Notes"]
    assert grids[0].rows[0] == ["grade", "step", "salary"]
    assert grids[0].rows[1] == ["GS-12", 5, 86415]


def test_read_sheets_csv_returns_single_grid(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("grade,step,salary\nGS-12,5,86415\n", encoding="utf-8")
    grids = read_sheets(p)
    assert len(grids) == 1
    assert grids[0].sheet_name == "data"
    assert grids[0].rows[0] == ["grade", "step", "salary"]
    assert grids[0].rows[1] == ["GS-12", "5", "86415"]  # csv cells are strings


@pytest.mark.parametrize("value,expected", [
    (None, "empty"),
    ("", "empty"),
    ("   ", "empty"),
    (5, "number"),
    (3.14, "number"),
    ("86415", "number"),
    ("$86,415", "number"),
    ("12%", "number"),
    ("GS-12", "text"),
    ("grade", "text"),
])
def test_cell_kind(value, expected):
    assert _cell_kind(value) == expected


def test_header_is_first_row_when_clean():
    rows = [["grade", "step", "salary"], ["GS-12", 5, 86415], ["GS-13", 5, 102000]]
    assert detect_header_row(rows) == 0


def test_header_after_title_and_blank_rows():
    rows = [
        ["2024 General Schedule"],     # title banner
        [None, None, None],            # blank
        ["grade", "step", "salary"],   # real header at index 2
        ["GS-12", 5, 86415],
    ]
    assert detect_header_row(rows) == 2


def test_no_header_returns_negative_one():
    # All-numeric grid with no labels => no detectable header.
    rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert detect_header_row(rows) == -1


def test_empty_grid_returns_negative_one():
    assert detect_header_row([]) == -1


def test_infer_dtypes_per_column():
    rows = [
        ["grade", "step", "salary"],
        ["GS-12", 5, 86415],
        ["GS-13", 5, 102000],
        ["GS-14", 6, 120000],
    ]
    assert infer_column_dtypes(rows, header_row_index=0) == ["text", "number", "number"]


def test_infer_dtypes_dominant_kind_wins_with_one_outlier():
    rows = [
        ["amount"],
        [100],
        [200],
        ["N/A"],   # one text outlier in a numeric column
        [300],
    ]
    assert infer_column_dtypes(rows, header_row_index=0) == ["number"]


def test_infer_dtypes_all_empty_column_is_empty():
    rows = [["a", "b"], [1, None], [2, None]]
    assert infer_column_dtypes(rows, header_row_index=0) == ["number", "empty"]


def test_clean_table_is_classified_clean():
    rows = [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 20)]
    result = classify_sheet(SheetGrid("Pay", rows))
    assert isinstance(result, SheetClassification)
    assert result.route == "clean"
    assert result.header_row_index == 0
    assert result.column_dtypes == ["text", "number", "number"]


def test_too_few_rows_is_messy():
    rows = [["grade", "salary"], ["GS-12", 86415]]  # only 1 data row (< MIN_DATA_ROWS)
    result = classify_sheet(SheetGrid("Tiny", rows))
    assert result.route == "messy"


def test_no_header_is_messy():
    rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    result = classify_sheet(SheetGrid("Raw", rows))
    assert result.route == "messy"
    assert result.header_row_index == -1


def test_ragged_non_rectangular_is_messy():
    # Wildly varying row widths => not a rectangular table.
    rows = [
        ["a", "b", "c"],
        ["note spanning"],
        ["x", "y"],
        ["only one"],
        ["p", "q", "r", "s", "t"],
    ]
    result = classify_sheet(SheetGrid("Messy", rows))
    assert result.route == "messy"
