"""Word/Excel embedded image region collection and text enrich."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image


def _png_bytes(w=400, h=300, color=(40, 80, 160)) -> bytes:
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_docx_with_image(path: Path, png: bytes) -> None:
    from docx import Document
    from docx.shared import Inches

    doc = Document()
    doc.add_heading("Network Overview", level=1)
    doc.add_paragraph("This document describes the SD-WAN portal architecture.")
    # write temp image file for python-docx
    img_path = path.parent / "_tmp_fig.png"
    img_path.write_bytes(png)
    try:
        doc.add_picture(str(img_path), width=Inches(4))
    finally:
        img_path.unlink(missing_ok=True)
    doc.add_paragraph("See figure above for components.")
    doc.save(str(path))


def _make_xlsx_with_image(path: Path, png: bytes) -> None:
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage

    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws["A1"] = "Budget"
    ws["A2"] = 100
    img_path = path.parent / "_tmp_xl.png"
    img_path.write_bytes(png)
    try:
        xl_img = XLImage(str(img_path))
        ws.add_image(xl_img, "C2")
        wb.save(str(path))
    finally:
        img_path.unlink(missing_ok=True)


def test_extract_image_regions_docx(tmp_path):
    from src.ingestion.figure_extract import extract_image_regions_docx

    docx_path = tmp_path / "with_fig.docx"
    _make_docx_with_image(docx_path, _png_bytes(500, 400))
    regions = extract_image_regions_docx(docx_path)
    assert len(regions) >= 1
    assert regions[0].source == "docx"
    assert regions[0].width >= 400
    assert regions[0].image_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_extract_image_regions_xlsx(tmp_path):
    from src.ingestion.figure_extract import extract_image_regions_xlsx

    xlsx_path = tmp_path / "with_fig.xlsx"
    _make_xlsx_with_image(xlsx_path, _png_bytes(480, 320))
    regions = extract_image_regions_xlsx(xlsx_path)
    assert len(regions) >= 1
    assert regions[0].source == "xlsx"
    assert regions[0].width >= 300


def test_enrich_text_with_figures_docx(tmp_path, monkeypatch):
    from src.config import settings
    from src.ingestion.figure_extract import enrich_text_with_figures, ImageKind, ProseBlock
    from src.ingestion.tabular import SheetGrid

    monkeypatch.setattr(settings, "figure_extraction_enabled", True)
    docx_path = tmp_path / "net.docx"
    _make_docx_with_image(docx_path, _png_bytes(600, 400))

    fake_prose = ProseBlock(
        text="[Figure p.1 — network]\nIdentifiers:\n- portal.example.com\n[/Figure]",
        page=0,
    )

    def fake_process(regions, **kwargs):
        from src.ingestion.figure_extract import FigureEnrichmentResult
        return FigureEnrichmentResult(
            prose_blocks=[fake_prose],
            table_grids=[],
            figures_seen=len(regions),
            figures_used=1,
        )

    with patch("src.ingestion.figure_extract.process_image_regions", side_effect=fake_process):
        text, grids = enrich_text_with_figures(
            docx_path, "Body paragraph about the network.",
        )
    assert "Body paragraph" in text
    assert "## Embedded figures" in text
    assert "portal.example.com" in text
    assert grids == []


def test_enrich_text_disabled(tmp_path, monkeypatch):
    from src.config import settings
    from src.ingestion.figure_extract import enrich_text_with_figures

    monkeypatch.setattr(settings, "figure_extraction_enabled", False)
    docx_path = tmp_path / "x.docx"
    _make_docx_with_image(docx_path, _png_bytes())
    text, grids = enrich_text_with_figures(docx_path, "hello")
    assert text == "hello"
    assert grids == []


def test_dispatch_for_path(tmp_path):
    from src.ingestion.figure_extract import extract_image_regions_for_path

    docx_path = tmp_path / "a.docx"
    _make_docx_with_image(docx_path, _png_bytes(400, 300))
    regs = extract_image_regions_for_path(docx_path)
    assert len(regs) >= 1

    xlsx_path = tmp_path / "b.xlsx"
    _make_xlsx_with_image(xlsx_path, _png_bytes(400, 300))
    regs2 = extract_image_regions_for_path(xlsx_path)
    assert len(regs2) >= 1
