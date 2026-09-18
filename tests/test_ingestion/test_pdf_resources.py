"""Regression coverage for bounded PDF OCR and per-page resource cleanup."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pdfplumber
import pytest
from pypdf import PdfReader, PdfWriter

from src.ingestion import pdf_extract

FIXTURE = Path(__file__).parents[1] / "fixtures" / "pdf" / "two_page_table.pdf"


def install_ocr(monkeypatch, partition):
    # Exercise real PDF splitting without downloading/loading OCR models.
    module = ModuleType("unstructured.partition.pdf")
    module.partition_pdf = partition
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize("ocr_error", [False, True])
def test_ocr_receives_only_requested_page_and_closes_temporary_file(
    tmp_path, monkeypatch, ocr_error,
):
    source = tmp_path / "rotated.pdf"
    with PdfWriter() as writer:
        writer.append(FIXTURE)
        writer.pages[1].rotate(90)
        writer.pages[1].cropbox.lower_left = (10, 10)
        writer.write(source)
    original = source.read_bytes()
    expected = PdfReader(source).pages[1]
    inputs = []

    def partition(*, file, strategy, infer_table_structure, starting_page_number):
        inputs.append(file)
        selected = PdfReader(file)
        assert len(selected.pages) == 1
        assert selected.pages[0].extract_text() == expected.extract_text()
        assert selected.pages[0].rotation == expected.rotation
        assert selected.pages[0].cropbox == expected.cropbox
        assert strategy == "hi_res"
        assert infer_table_structure is True
        assert starting_page_number == 2
        if ocr_error:
            raise RuntimeError("OCR failed")
        return ["Selected page"]

    install_ocr(monkeypatch, partition)
    if ocr_error:
        with pytest.raises(RuntimeError, match="OCR failed"):
            pdf_extract._partition_scanned(source, 1)
    else:
        assert pdf_extract._partition_scanned(source, 1) == ["Selected page"]
    assert len(inputs) == 1 and inputs[0].closed
    assert source.read_bytes() == original


def test_multiple_scanned_pages_are_ocrd_once_with_original_page_numbers(monkeypatch):
    expected = [page.extract_text() for page in PdfReader(FIXTURE).pages]
    calls = []
    source_readers = []

    def recording_reader(stream):
        reader = PdfReader(stream)
        source_readers.append(reader)
        return reader

    class TextElement:
        category = "NarrativeText"

        def __init__(self, text):
            self.text = text

        def __str__(self):
            return self.text

    def partition(*, file, starting_page_number, **kwargs):
        assert all(reader.stream.closed and not reader.resolved_objects for reader in source_readers)
        pages = PdfReader(file).pages
        assert len(pages) == 1
        text = pages[0].extract_text()
        calls.append((starting_page_number, text))
        return [TextElement(text)]

    install_ocr(monkeypatch, partition)
    monkeypatch.setattr("pypdf.PdfReader", recording_reader)
    monkeypatch.setattr(pdf_extract, "DIGITAL_MIN_CHARS", 10**9)
    result = pdf_extract.extract_pdf(FIXTURE)

    assert result.method == "ocr"
    assert calls == list(enumerate(expected, start=1))
    assert [block.page for block in result.prose_blocks] == [0, 1]
    assert [block.text for block in result.prose_blocks] == [text.strip() for text in expected]
    assert len(source_readers) == len(expected)
    assert all(reader.stream.closed and not reader.resolved_objects for reader in source_readers)


@pytest.mark.parametrize("mode", ["digital", "ocr", "failure"])
def test_page_caches_released_before_next_page_ocr_and_document_close(monkeypatch, mode):
    real_open = pdfplumber.open
    visited = []

    def assert_released(pages):
        for page in pages:
            assert "_layout" not in page.__dict__
            assert "_objects" not in page.__dict__

    def tracked_open(*args, **kwargs):
        pdf = real_open(*args, **kwargs)
        real_close = pdf.close

        def close():
            assert_released(visited)
            real_close()

        monkeypatch.setattr(pdf, "close", close)
        for page in pdf.pages:
            extract_text = page.extract_text

            def text(*args, _page=page, _extract=extract_text, **kwargs):
                assert_released(p for p in visited if p is not _page)
                if _page not in visited:
                    visited.append(_page)
                value = _extract(*args, **kwargs)
                assert "_layout" in _page.__dict__
                assert "_objects" in _page.__dict__
                return value

            monkeypatch.setattr(page, "extract_text", text)
        return pdf

    monkeypatch.setattr(pdfplumber, "open", tracked_open)
    if mode == "ocr":
        monkeypatch.setattr(pdf_extract, "DIGITAL_MIN_CHARS", 10**9)

        def partition(**kwargs):
            assert_released(visited)
            return []

        install_ocr(monkeypatch, partition)
    if mode == "failure":
        def fail(*args):
            raise ValueError("Bad table")

        monkeypatch.setattr(pdf_extract, "_page_tables", fail)
        with pytest.raises(ValueError, match="Bad table"):
            pdf_extract.extract_pdf(FIXTURE)
    else:
        pdf_extract.extract_pdf(FIXTURE)
    assert len(visited) == (1 if mode == "failure" else 2)


@pytest.mark.parametrize("borderless", [False, True])
@pytest.mark.parametrize("line_failure", [False, True])
def test_table_detection_reused_for_prose_and_layout(monkeypatch, borderless, line_failure):
    calls = []
    real_find = pdfplumber.page.Page.find_tables

    def find(page, table_settings=None):
        calls.append((page.page_number, table_settings))
        if borderless and table_settings is None:
            return []
        # Use the fixture's ruled grid as a deterministic text-strategy result.
        return real_find(page)

    monkeypatch.setattr(pdfplumber.page.Page, "find_tables", find)
    if line_failure:
        def fail(*args, **kwargs):
            raise ValueError("Text lines unavailable")

        monkeypatch.setattr(pdfplumber.page.Page, "extract_text_lines", fail)
    result = pdf_extract.extract_pdf(FIXTURE)
    assert any("O-1" in str(grid.rows) for grid in result.table_grids)
    assert all("O-1" not in block.text for block in result.prose_blocks + result.layout_blocks)
    expected = []
    for number in (1, 2):
        expected.append((number, None))
        if borderless:
            expected.append((number, pdf_extract._TABLE_SETTINGS))
    assert calls == expected


def test_ocr_memory_exhaustion_propagates(monkeypatch):
    def fail(*args, **kwargs):
        raise MemoryError("OCR budget exhausted")

    monkeypatch.setattr(pdf_extract, "_partition_scanned", fail)
    with pytest.raises(MemoryError, match="OCR budget exhausted"):
        pdf_extract._extract_scanned_page(FIXTURE, 0)


def test_layout_memory_exhaustion_propagates():
    def fail(**kwargs):
        raise MemoryError("Layout budget exhausted")

    page = SimpleNamespace(extract_text_lines=fail)
    with pytest.raises(MemoryError, match="Layout budget exhausted"):
        pdf_extract._page_layout_lines(page, 0, [])
