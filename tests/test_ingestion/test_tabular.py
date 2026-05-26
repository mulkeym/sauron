"""Tests for src/ingestion/tabular.py — structured parse + clean/messy classification."""
from pathlib import Path

import openpyxl
import pytest

from src.ingestion.tabular import SheetGrid, read_sheets
from src.ingestion.tabular import _cell_kind


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
