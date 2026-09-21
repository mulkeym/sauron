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


def test_digital_pdf_exposes_positioned_text_lines():
    sample = Path("test_fixtures/sample.pdf")
    result = extract_pdf(sample)
    assert result.layout_blocks
    assert all(block.bbox is not None for block in result.layout_blocks)
    assert all(block.page == 0 for block in result.layout_blocks)
    assert any("Expense Reporting" in block.text for block in result.layout_blocks)


def test_borderless_table_inference_does_not_slice_procedure_words():
    from pathlib import Path
    result = extract_pdf(Path(__file__).parents[1] / "fixtures/pdf/prose_procedure.pdf")
    text = " ".join(" ".join(block.text for block in result.prose_blocks).split())
    assert not result.table_grids
    assert "If the status is WAITING, check whether the maintenance window is active." in text
    assert "A missing window means the deployment should be deferred." in text
    assert "status READY and route tag 417" in text


def test_borderless_columns_between_words_are_not_rejected():
    from types import SimpleNamespace
    from src.ingestion.pdf_extract import _text_table_splits_words
    table = SimpleNamespace(cells=[(0, 0, 50, 20), (50, 0, 100, 20)], bbox=(0, 0, 100, 20))
    words = [dict(x0=5, x1=40, top=1, bottom=10), dict(x0=60, x1=95, top=1, bottom=10)]
    assert not _text_table_splits_words(SimpleNamespace(extract_words=lambda: words), table)
    words += [dict(x0=45, x1=65, top=1, bottom=10), dict(x0=40, x1=60, top=11, bottom=18)]
    assert _text_table_splits_words(SimpleNamespace(extract_words=lambda: words), table)


def _sheared_caption_line(text):
    """Position-only word spaces and overlapping synthetic-italic glyph boxes."""
    from pdfplumber.utils import extract_text
    chars, pen = [], 20.0
    for character in text:
        if character == ' ':
            pen += 1.776
            continue
        chars.append(dict(text=character, matrix=(1., 0., .2126, 1., pen, 100.),
                          adv=3., upright=True, size=8., x0=pen, x1=pen + 4.7008,
                          top=10., bottom=18., doctop=10., width=4.7008, height=8.))
        pen += 3.
    return dict(text=extract_text(chars), chars=chars, x0=20., x1=pen, top=10., bottom=18.)


@pytest.mark.parametrize('caption', [
    'Figure 3: Cisco Catalyst SD-WAN Portal Architecture',
    'Figure 2: Cisco Catalyst SD-WAN Portal Benefits and Operations',
    'Figure 13: QoS Information for the google-services Application',
    'Figure 7: IPv6 vManage eBGP API2 and iPhone',
])
def test_sheared_caption_uses_baseline_spacing(caption):
    from src.ingestion.pdf_extract import _layout_line_text, _page_layout_lines
    from types import SimpleNamespace
    line = _sheared_caption_line(caption)
    assert line['text'] == caption.replace(' ', '')  # reproduce old extraction
    assert _layout_line_text(line) == caption
    page = SimpleNamespace(extract_text_lines=lambda **kwargs: [line])
    block, = _page_layout_lines(page, 22, [])
    assert block.text == caption and block.page == 22
    assert block.bbox == (20., 10., line['x1'], 18.)


def test_caption_spacing_preserves_existing_spaces_and_source_characters():
    from src.ingestion.pdf_extract import _layout_line_text
    line = _sheared_caption_line('Figure 1: SD-WAN QoS')
    line['text'] = 'Figure 1: SD-WAN QoS'
    assert _layout_line_text(line) == line['text']
    line['text'] += ' additional source text'
    assert _layout_line_text(line) == line['text']


@pytest.mark.parametrize('change', ['missing_matrix', 'rotated', 'not_sheared'])
def test_caption_spacing_leaves_unsupported_geometry_unchanged(change):
    from src.ingestion.pdf_extract import _layout_line_text
    line = _sheared_caption_line('Figure 1: SD-WAN')
    for c in line['chars']:
        if change == 'missing_matrix':
            c.pop('matrix')
        elif change == 'rotated':
            c['matrix'] = (0., 1., .2, 0., c['x0'], 100.)
        else:
            c['matrix'] = (1., 0., 0., 1., c['x0'], 100.)
    assert _layout_line_text(line) == line['text']
