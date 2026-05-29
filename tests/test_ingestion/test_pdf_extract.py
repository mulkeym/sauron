from src.ingestion.pdf_extract import normalize_grid
from src.ingestion.tabular import SheetGrid


def test_normalize_grid_drops_empty_rows_cols_and_fills_none():
    raw = [
        ["Grade", "Over 2", None, ""],
        [None, None, None, None],          # fully-empty row -> dropped
        ["O-1", "3998.40", None, ""],
    ]
    grid = normalize_grid(raw, sheet_name="p1_table1")
    assert isinstance(grid, SheetGrid)
    assert grid.sheet_name == "p1_table1"
    assert grid.rows == [["Grade", "Over 2"], ["O-1", "3998.40"]]
