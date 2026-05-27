"""Tests for src/ingestion/tabular_chunker.py — sheet-aware chunking + region narratives."""
from src.ingestion.chunker import Chunk
from src.ingestion.tabular import SheetGrid, SheetClassification
from src.ingestion.tabular_chunker import structure_aware_chunks


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
