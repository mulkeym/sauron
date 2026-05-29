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


from src.ingestion.pdf_extract import stitch_tables


def test_stitch_merges_consecutive_tables_with_matching_header():
    g1 = SheetGrid("p1_table1", [["Grade", "Over 2"], ["O-1", "3998"]])
    g2 = SheetGrid("p2_table1", [["Grade", "Over 2"], ["E-1", "2017"]])  # same header
    g3 = SheetGrid("p3_table1", [["Loc", "Pct"], ["RUS", "16.5"]])       # different header
    out = stitch_tables([g1, g2, g3])
    assert len(out) == 2
    assert out[0].rows == [["Grade", "Over 2"], ["O-1", "3998"], ["E-1", "2017"]]
    assert out[1].rows == [["Loc", "Pct"], ["RUS", "16.5"]]


from src.ingestion.pdf_extract import grid_width_consistent


def test_grid_width_consistent_detects_ragged_data():
    good = SheetGrid("t", [["Grade", "A", "B"], ["O-1", "1", "2"], ["E-1", "3", "4"]])
    bad = SheetGrid("t", [["Grade", "A", "B"], ["O-1", "1"], ["E-1", "3", "4", "5"]])
    assert grid_width_consistent(good) is True
    assert grid_width_consistent(bad) is False


def test_grid_width_consistent_trivially_true_for_tiny_grid():
    assert grid_width_consistent(SheetGrid("t", [["only one row"]])) is True
