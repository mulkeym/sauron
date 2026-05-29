from pathlib import Path
import pytest
from src.ingestion.pdf_extract import extract_pdf

FIX = Path("tests/fixtures/pdf/two_page_table.pdf")


@pytest.mark.skipif(not FIX.exists(), reason="fixture missing")
def test_extract_pdf_digital_returns_grids_and_prose():
    result = extract_pdf(FIX)
    assert result.method in ("digital", "mixed")
    all_cells = [c for g in result.table_grids for row in g.rows for c in row]
    assert any("O-1" in c for c in all_cells)
    assert any("E-3" in c for c in all_cells)
    # multi-page same-header tables stitched into one grid
    grades = [g for g in result.table_grids if g.rows and "Grade" in g.rows[0][0]]
    assert len(grades) == 1


def test_extract_scanned_page_parses_html_tables(monkeypatch):
    from src.ingestion import pdf_extract

    class _Meta:
        def __init__(self, html=None): self.text_as_html = html

    class _El:
        def __init__(self, cat, text, html=None):
            self.category = cat
            self._text = text
            self.metadata = _Meta(html)
        def __str__(self): return self._text

    html = "<table><tr><td>Grade</td><td>Pay</td></tr><tr><td>E-1</td><td>2017</td></tr></table>"
    fake = [
        _El("Title", "Active Duty Pay"),
        _El("NarrativeText", "Monthly basic pay follows."),
        _El("Table", "Grade Pay E-1 2017", html),
    ]
    monkeypatch.setattr(pdf_extract, "_partition_scanned", lambda path, page_no: fake)

    blocks, grids = pdf_extract._extract_scanned_page(FIX, 0)
    assert any("Monthly basic pay" in b.text for b in blocks)
    assert len(grids) == 1
    assert ["E-1", "2017"] in grids[0].rows
