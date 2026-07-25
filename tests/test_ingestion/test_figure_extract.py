"""Unit tests for embedded figure extraction (OCR / table / vision strategies)."""
from __future__ import annotations

import io
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from src.ingestion.figure_extract import (
    ImageKind,
    ImageRegion,
    classify_region,
    merge_prose_by_page,
    parse_markdown_table,
    strategy_table,
    _should_skip_size,
    enrich_pdf_with_figures,
)
from src.ingestion.pdf_extract import ExtractedPdf, ProseBlock
from src.config import settings


def _png_bytes(w=200, h=200, color=(200, 200, 200)) -> bytes:
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_parse_markdown_table_basic():
    md = """
| Host | IP |
|------|-----|
| edge-01 | 10.0.0.1 |
| edge-02 | 10.0.0.2 |
"""
    rows = parse_markdown_table(md)
    assert rows is not None
    assert len(rows) == 3
    assert rows[0] == ["Host", "IP"]
    assert rows[1][1] == "10.0.0.1"


def test_parse_markdown_table_not_a_table():
    assert parse_markdown_table("NOT_A_TABLE") is None
    assert parse_markdown_table("just some prose") is None


def test_parse_csv_fallback():
    text = "vlan,subnet\n10,10.10.10.0/24\n20,10.10.20.0/24"
    rows = parse_markdown_table(text)
    assert rows is not None
    assert rows[0] == ["vlan", "subnet"]
    assert len(rows) == 3


def test_should_skip_tiny_images():
    assert _should_skip_size(20, 20) is True
    assert _should_skip_size(400, 300) is False
    assert _should_skip_size(2000, 10) is True  # extreme aspect


def test_classify_table_like_ocr():
    ocr = "\n".join([
        "VLAN   Subnet          Gateway",
        "10     10.10.10.0/24   10.10.10.1",
        "20     10.10.20.0/24   10.10.20.1",
        "30     10.10.30.0/24   10.10.30.1",
    ])
    r = ImageRegion(
        page=0, index=0, image_bytes=b"x", width=600, height=400, ocr_text=ocr,
    )
    assert classify_region(r) == ImageKind.TABLE


def test_classify_text_scan():
    ocr = (
        "This is a scanned paragraph of documentation about change management "
        "procedures for the operations team and related policy requirements.\n"
        "Another full sentence follows with more descriptive language only about "
        "how the system should be operated in production environments carefully."
    )
    r = ImageRegion(
        page=0, index=0, image_bytes=b"x", width=600, height=400, ocr_text=ocr,
    )
    assert classify_region(r) == ImageKind.TEXT_SCAN


def test_classify_network_ips():
    ocr = "vManage 10.1.1.1\nvBond 10.1.1.2\nvSmart 10.1.1.3"
    r = ImageRegion(
        page=0, index=0, image_bytes=b"x", width=800, height=600, ocr_text=ocr,
    )
    kind = classify_region(r)
    assert kind in (ImageKind.NETWORK, ImageKind.PROCESS)


def test_classify_portal_diagram_not_text_scan():
    """p.22-style multi-panel marketing diagram must not take OCR-only path."""
    ocr = "\n".join([
        "Controller Lifecycle Management",
        "Just-in-time provisioning",
        "Region selection",
        "Cloud provider selection",
        "Visibility & Regulation",
        "Cloud infrastructure",
        "Holistic device status",
        "Deployment Accelerator",
        "Simple day-0 cloud operation",
        "Operation Services",
        "RBAC",
        "Backup",
        "TACACS",
        "Cisco SD-WAN Overlay Network",
    ])
    r = ImageRegion(
        page=21, index=0, image_bytes=b"x", width=1217, height=526, ocr_text=ocr,
    )
    kind = classify_region(r)
    assert kind != ImageKind.TEXT_SCAN, f"got {kind}"
    assert kind in (ImageKind.PROCESS, ImageKind.NETWORK), f"got {kind}"


def test_merge_ocr_into_vision_appends_missing_high_value():
    from src.ingestion.figure_extract import merge_ocr_into_vision
    vision = "[Figure p.22 — process]\nLayout:\n- center portal\n[/Figure]"
    ocr = "TACACS RBAC https://ssp-gov.sdwan.example.com Deployment Accelerator"
    out = merge_ocr_into_vision(vision, ocr, page=21, kind="process")
    assert "TACACS" in out
    assert "https://ssp-gov.sdwan.example.com" in out
    assert "Also visible (OCR" in out
    assert out.count("[/Figure]") == 1


def test_merge_ocr_skips_when_vision_covers_labels():
    """p.22 case: good vision → no garble OCR appendix."""
    from src.ingestion.figure_extract import merge_ocr_into_vision, select_ocr_tokens_to_merge
    vision = """[Figure p.22 — network]
Identifiers:
- Cisco SD-WAN Self-Service Portal
- Cisco SD-WAN Overlay Network
Nodes:
- Controller Lifecycle Management
- Visibility & Regulation
- Deployment Accelerator
- Operation Services
Notes:
- Just-in-time provisioning, Region selection, Cloud provider selection
- RBAC, Backup, TACACS
[/Figure]"""
    ocr = """Controller Lifecycle Management e
Visibility & Regulation
Just-in-time provisionin:
Pisco eo WAN
Cloud provider selection NEE ee
Cisco SD-WAN py Sa EN, SEU
Overlay Network Overlay Network
Deployment Accelerator i] & Operation Services
peg La Ve ted TOOLS
TACACS
RBAC
"""
    missing = select_ocr_tokens_to_merge(vision, ocr)
    assert missing == [], f"expected no merge tokens, got {missing}"
    out = merge_ocr_into_vision(vision, ocr, page=21, kind="network")
    assert "Also visible" not in out
    assert "Pisco" not in out
    assert "Self-Service Portal" in out


def test_merge_ocr_keeps_missing_url():
    from src.ingestion.figure_extract import select_ocr_tokens_to_merge
    vision = "[Figure p.23 — network]\nNodes:\n- ALB\n- Okta\n[/Figure]"
    ocr = "https://vorchestrator-gov.sdwan.cisco.com ALB Okta WAF"
    keep = select_ocr_tokens_to_merge(vision, ocr)
    assert any("vorchestrator" in t for t in keep)
    assert not any(t == "ALB" or t == "Okta" for t in keep)  # already in vision


def test_is_ocr_garble():
    from src.ingestion.figure_extract import is_ocr_garble, is_high_value_ocr_token
    assert is_ocr_garble("Pisco eo WAN") or not is_high_value_ocr_token("Pisco eo WAN")
    assert is_ocr_garble("peg La Ve ted TOOLS") or not is_high_value_ocr_token("peg La Ve ted TOOLS")
    assert is_high_value_ocr_token("TACACS")
    assert is_high_value_ocr_token("https://ssp-gov.sdwan.example.com")
    assert is_high_value_ocr_token("10.1.1.0/24") or is_high_value_ocr_token("10.1.1.1")


def test_skip_small_square_icons():
    from src.ingestion.figure_extract import _should_skip_size
    assert _should_skip_size(136, 136) is True
    assert _should_skip_size(1217, 526) is False


def test_looks_like_refusal():
    from src.ingestion.figure_extract import looks_like_refusal
    assert looks_like_refusal(
        "I cannot fulfill this request. The image is a cartoon squirrel."
    )
    assert not looks_like_refusal(
        "[Figure p.1 — illustration]\nSubject:\n- A fox in autumn woods\n[/Figure]"
    )


def test_strategy_process_falls_back_to_illustration():
    from src.ingestion.figure_extract import strategy_process, ImageRegion
    region = ImageRegion(
        page=0, index=0, image_bytes=_png_bytes(800, 600),
        width=800, height=600, ocr_text="", kind=None,
    )
    refusal = "I cannot fulfill this request. Not a technical diagram."
    illustration = (
        "[Figure p.1 — illustration]\nSubject:\n- Cartoon squirrel\n"
        "Setting / scene:\n- Autumn forest\n[/Figure]"
    )
    with patch("src.ingestion.figure_extract._vision") as mock_v:
        mock_v.side_effect = [refusal, illustration]
        grids, prose = strategy_process(region)
    assert grids == []
    assert prose and "squirrel" in prose[0].text.lower()
    assert "illustration" in prose[0].text.lower()
    assert mock_v.call_count == 2


def test_merge_prose_by_page_interleaves():
    base = [
        ProseBlock(text="page0 prose", page=0),
        ProseBlock(text="page2 prose", page=2),
    ]
    extra = [
        ProseBlock(text="fig page1", page=1),
        ProseBlock(text="fig page0", page=0),
    ]
    merged = merge_prose_by_page(base, extra)
    pages = [b.page for b in merged]
    assert pages == sorted(pages)
    # page 0 prose before/with page 0 figure
    assert merged[0].page == 0
    assert any(b.text == "fig page1" for b in merged)


def test_strategy_table_uses_vision_and_builds_grid():
    md = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    region = ImageRegion(
        page=1, index=0, image_bytes=_png_bytes(), width=200, height=200, ocr_text="",
        kind=ImageKind.TABLE,
    )
    with patch("src.ingestion.figure_extract._vision", return_value=md):
        grids, prose = strategy_table(region)
    assert len(grids) == 1
    assert grids[0].sheet_name == "p1_imgtable0"
    assert len(grids[0].rows) >= 2
    assert prose and "table" in prose[0].text.lower()


def test_strategy_table_fallback_on_not_a_table():
    region = ImageRegion(
        page=0, index=0, image_bytes=_png_bytes(), width=200, height=200,
        ocr_text="hello world from ocr text enough chars",
        kind=ImageKind.TABLE,
    )
    with patch("src.ingestion.figure_extract._vision", return_value="NOT_A_TABLE"):
        grids, prose = strategy_table(region)
    assert grids == []
    assert prose  # OCR fallback


def test_enrich_disabled_returns_same(monkeypatch):
    monkeypatch.setattr(settings, "figure_extraction_enabled", False)
    extracted = ExtractedPdf(
        prose_blocks=[ProseBlock(text="hi", page=0)],
        table_grids=[],
        method="digital",
    )
    out = enrich_pdf_with_figures("dummy.pdf", extracted)
    assert out is extracted or out.prose_blocks == extracted.prose_blocks


def test_enrich_merges_mocked_regions(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "figure_extraction_enabled", True)
    monkeypatch.setattr(settings, "figure_max_per_doc", 5)
    monkeypatch.setattr(settings, "figure_ocr_first", True)

    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    region = ImageRegion(
        page=0, index=0, image_bytes=_png_bytes(300, 300),
        width=300, height=300, source="embedded",
        ocr_text="Host  IP\nedge  10.0.0.1\ncore  10.0.0.2",
    )
    extracted = ExtractedPdf(
        prose_blocks=[ProseBlock(text="Intro paragraph", page=0)],
        table_grids=[],
        method="digital",
    )
    md = "| Host | IP |\n|---|---|\n| edge | 10.0.0.1 |\n| core | 10.0.0.2 |"

    with patch("src.ingestion.figure_extract.extract_image_regions", return_value=[region]), \
         patch("src.ingestion.figure_extract.ocr_image", return_value=region.ocr_text), \
         patch("src.ingestion.figure_extract._vision", return_value=md), \
         patch("src.ingestion.figure_extract._vision_classify", return_value=None):
        # Force TABLE classification
        with patch("src.ingestion.figure_extract.classify_region", return_value=ImageKind.TABLE):
            out = enrich_pdf_with_figures(pdf_path, extracted)

    assert len(out.table_grids) == 1
    assert any("table" in b.text.lower() or "Figure" in b.text for b in out.prose_blocks)
    assert any(b.text == "Intro paragraph" for b in out.prose_blocks)
