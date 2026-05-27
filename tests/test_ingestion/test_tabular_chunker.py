"""Tests for src/ingestion/tabular_chunker.py — sheet-aware chunking + region narratives."""
from src.ingestion.chunker import Chunk
from src.ingestion.tabular import SheetGrid, SheetClassification
from src.ingestion.tabular_chunker import (
    structure_aware_chunks, find_table_region, messy_region_narratives,
    sheets_needing_text, build_tier_chunks,
)


def test_structure_aware_chunks_repeats_header_and_marks_sheet():
    rows = [["grade", "step", "salary"], ["GS-12", "5", "86415"], ["GS-13", "1", "90000"]]
    chunks = structure_aware_chunks("Pay", rows, header_row_index=0, chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Sheet: Pay\ngrade | step | salary")
    assert "GS-12 | 5 | 86415" in chunks[0].text
    assert "GS-13 | 1 | 90000" in chunks[0].text


def test_structure_aware_chunks_never_splits_mid_row():
    # 6 data rows; a tiny chunk_size forces multiple chunks, each carrying the header.
    rows = [["a", "b"]] + [[f"r{i}", f"v{i}"] for i in range(6)]
    chunks = structure_aware_chunks("S", rows, header_row_index=0, chunk_size=30)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith("Sheet: S\na | b")
        # every non-header line is a whole "rN | vN" row, never a fragment
        body_lines = c.text.split("\n")[2:]
        for line in body_lines:
            assert line.count("|") == 1
            assert line.split(" | ")[0].startswith("r")


def test_structure_aware_chunks_no_header_emits_all_rows_with_marker():
    rows = [["intro note"], ["x", "y"], ["1", "2"]]
    chunks = structure_aware_chunks("Mess", rows, header_row_index=-1, chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Sheet: Mess\n")
    assert "intro note" in chunks[0].text
    assert "x | y" in chunks[0].text


def test_structure_aware_chunks_indices_and_start_offsets():
    rows = [["a", "b"]] + [[f"r{i}", f"v{i}"] for i in range(6)]
    chunks = structure_aware_chunks("S", rows, header_row_index=0, chunk_size=30,
                                    start_index=10, start_char=500)
    assert chunks[0].index == 10
    assert chunks[1].index == 11
    assert chunks[0].start_char == 500
    assert chunks[1].start_char == 500 + len(chunks[0].text) + 1


def test_find_table_region_in_messy_sheet():
    # title banner + blank-ish preamble, then a clean rectangular block, then trailing note
    rows = [
        ["2026 GS Pay Table"],            # 0 single-cell banner
        ["grade", "step", "salary"],      # 1 header
        ["GS-12", "5", "86415"],          # 2 data
        ["GS-13", "1", "90000"],          # 3 data
        ["GS-14", "2", "99000"],          # 4 data
        ["note: rates effective Jan 1"],  # 5 trailing (width mismatch ends region)
    ]
    region = find_table_region(rows)
    assert region == (1, 5)  # header at row 1, data rows [2,5)


def test_find_table_region_returns_none_when_no_block():
    rows = [["just"], ["some"], ["prose"], ["lines"]]
    assert find_table_region(rows) is None


def test_find_table_region_requires_min_data_rows():
    rows = [["a", "b"], ["1", "2"]]  # only one data row beneath header
    assert find_table_region(rows) is None


def test_messy_region_narratives_restates_rows_no_llm():
    rows = [
        ["2026 GS Pay Table"],
        ["locality", "grade", "salary"],
        ["Tampa", "GS-12", "86415"],
        ["Boston", "GS-12", "92000"],
        ["Denver", "GS-13", "99000"],
    ]
    grid = SheetGrid(sheet_name="Pay", rows=rows)
    region = find_table_region(rows)
    assert region is not None
    narratives = messy_region_narratives(grid, region)
    assert len(narratives) == 3  # one per data row in the region
    # keys (text cols) as context, measures (number cols) restated; raw values, no math
    joined = " ".join(narratives)
    assert "Tampa" in joined and "86415" in joined
    assert "Pay" in narratives[0]  # context defaults to sheet name


def test_messy_region_narratives_missing_cell_not_fabricated():
    rows = [
        ["locality", "grade", "salary"],
        ["Tampa", "GS-12", "86415"],
        ["Boston", "GS-12"],            # missing salary cell
        ["Denver", "GS-13", "99000"],
    ]
    grid = SheetGrid(sheet_name="Pay", rows=rows)
    region = find_table_region(rows)
    narratives = messy_region_narratives(grid, region)
    assert any("(not specified)" in n for n in narratives)


def _grid(name, rows):
    return SheetGrid(sheet_name=name, rows=rows)


def _cls(name, route, hdr=0):
    return SheetClassification(sheet_name=name, route=route, header_row_index=hdr)


def test_sheets_needing_text_skips_ingested_clean_keeps_messy_and_failed_clean():
    grids = [_grid("Clean", [["a", "b"]]), _grid("Messy", [["x"]]), _grid("Failed", [["c", "d"]])]
    clss = [_cls("Clean", "clean"), _cls("Messy", "messy", hdr=-1), _cls("Failed", "clean")]
    ingested = {"Clean"}  # only "Clean" structured-ingest succeeded
    out = sheets_needing_text(grids, clss, ingested)
    names = [g.sheet_name for g, _ in out]
    assert names == ["Messy", "Failed"]  # ingested clean dropped; messy + failed-clean kept


def test_build_tier_chunks_concatenates_across_sheets_with_continuous_indices():
    grids = [
        _grid("A", [["h1", "h2"], ["a", "1"], ["b", "2"]]),
        _grid("B", [["h3", "h4"], ["c", "3"], ["d", "4"]]),
    ]
    clss = [_cls("A", "messy"), _cls("B", "messy")]
    text_sheets = list(zip(grids, clss))
    chunks = build_tier_chunks(text_sheets, chunk_size=1000)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert any("Sheet: A" in c.text for c in chunks)
    assert any("Sheet: B" in c.text for c in chunks)


def test_build_tier_chunks_empty_when_no_sheets():
    assert build_tier_chunks([], chunk_size=1000) == []
