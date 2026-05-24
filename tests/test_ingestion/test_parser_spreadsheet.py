"""Tests for spreadsheet / binary parsing in the ingestion parser.

Regression guard for the bug where legacy binary .xls files fell through to
the plaintext branch and were ingested as OLE-binary mojibake.
"""
import csv
from pathlib import Path

import pytest

from src.ingestion.parser import (
    _parse_plaintext,
    _parse_spreadsheet,
    parse_document,
)

FIXTURES = Path(__file__).resolve().parents[1] / ".." / "test_fixtures"


def test_xls_extension_routes_to_spreadsheet_parser():
    # .xls must not fall through to plaintext (which would yield binary garbage)
    sample = (FIXTURES / "sample.xls").resolve()
    pytest.importorskip("xlrd")
    parsed = parse_document(sample)
    assert parsed.doc_type == "xls"
    assert "\x00" not in parsed.text
    assert "Sheet:" in parsed.text


def test_legacy_xls_extracts_cell_values():
    pytest.importorskip("xlrd")
    sample = (FIXTURES / "sample.xls").resolve()
    parsed = _parse_spreadsheet(sample)
    # Cell contents from the fixture must appear as readable text
    assert "Grade" in parsed.text
    assert "GS-1" in parsed.text
    assert "20172" in parsed.text


def test_xlsx_still_parses(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    p = tmp_path / "rates.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Grade", "Step 1"])
    ws.append(["GS-1", 20172])
    wb.save(p)
    parsed = _parse_spreadsheet(p)
    assert parsed.doc_type == "xlsx"
    assert "Grade" in parsed.text
    assert "20172" in parsed.text


def test_csv_parses_as_delimited(tmp_path):
    p = tmp_path / "data.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Grade", "Step 1"])
        w.writerow(["GS-1", "20172"])
    parsed = _parse_spreadsheet(p)
    assert "Grade | Step 1" in parsed.text
    assert "GS-1 | 20172" in parsed.text


def test_tsv_parses_as_delimited(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("Grade\tStep 1\nGS-1\t20172\n")
    parsed = _parse_spreadsheet(p)
    assert "Grade | Step 1" in parsed.text


def test_plaintext_rejects_binary(tmp_path):
    p = tmp_path / "weird.conf"
    p.write_bytes(b"some text\x00\x01\x02 binary content")
    with pytest.raises(ValueError, match="binary"):
        _parse_plaintext(p)


def test_plaintext_accepts_real_text(tmp_path):
    p = tmp_path / "app.log"
    p.write_text("2021-01-01 INFO starting up\n")
    parsed = _parse_plaintext(p)
    assert "starting up" in parsed.text
