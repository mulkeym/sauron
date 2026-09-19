"""Extraction inside the disposable worker, with a typed JSON result."""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class PreparedDocument:
    parsed: ParsedDocument
    spreadsheet_grids: list = field(default_factory=list)
    pdf: object = None
    office: object = None
    warnings: list[str] = field(default_factory=list)
    figure_staging: str = ""


async def prepare_document(path: Path, filename: str, progress_cb=None) -> PreparedDocument:
    """Read all source-file formats here; indexing consumes only the result.

    Ordinary extraction exceptions retain the existing flat-text fallback.
    Process death, timeout and memory-limit failures never fall back to an
    unisolated parser in the API.
    """
    from src.config import settings
    from src.ingestion.parser import parse_document
    from src.ingestion.pdf_extract import extract_pdf
    from src.ingestion.tabular import read_sheets
    from src.ingestion.figure_extract import (
        enrich_pdf_with_figures_async, enrich_office_document_with_figures_async,
    )

    def progress(message):
        if progress_cb:
            progress_cb(message)

    progress(f"Parsing {filename}")
    parsed = parse_document(path)
    parsed.filename = filename
    result = PreparedDocument(parsed)

    def warning(stage, exc):
        message = f"{stage} failed: {type(exc).__name__}: {str(exc)[:500]}"
        result.warnings.append(message)
        logger.warning(message)

    if parsed.doc_type == "vsdx":
        from src.ingestion.visio import render_visio
        result.office, notices = render_visio(path, parsed, progress)
        result.warnings.extend(notices)
    elif parsed.doc_type in ("xlsx", "xls", "csv", "tsv"):
        progress("Reading spreadsheet tables")
        try:
            result.spreadsheet_grids = read_sheets(path)
        except MemoryError:
            raise
        except Exception as exc:
            warning("Spreadsheet table extraction", exc)
        if settings.figure_extraction_enabled and path.suffix.lower() in (".xlsx", ".xlsm"):
            try:
                result.office = await enrich_office_document_with_figures_async(path, "", progress_cb=progress)
            except MemoryError:
                raise
            except Exception as exc:
                warning("Spreadsheet figure extraction", exc)
    elif parsed.doc_type == "pdf":
        progress("Extracting PDF tables")
        try:
            result.pdf = extract_pdf(path)
        except MemoryError:
            raise
        except Exception as exc:
            warning("PDF table extraction; using flat text", exc)
        if result.pdf is not None and settings.figure_extraction_enabled:
            progress("Extracting PDF figures and OCR")
            try:
                result.pdf = await enrich_pdf_with_figures_async(path, result.pdf, progress_cb=progress)
            except MemoryError:
                raise
            except Exception as exc:
                warning("PDF figure extraction", exc)
    elif parsed.doc_type in ("docx", "pptx") and settings.figure_extraction_enabled:
        progress("Extracting document figures")
        try:
            result.office = await enrich_office_document_with_figures_async(
                path, parsed.text or "", document_blocks=parsed.blocks, progress_cb=progress,
            )
        except MemoryError:
            raise
        except Exception as exc:
            warning("Office figure extraction", exc)
    return result


def encode_prepared(value):
    if dataclasses.is_dataclass(value):
        return {"$type": type(value).__name__, "fields": {
            f.name: encode_prepared(getattr(value, f.name)) for f in dataclasses.fields(value)
        }}
    if isinstance(value, tuple):
        return {"$type": "tuple", "fields": [encode_prepared(v) for v in value]}
    if isinstance(value, list):
        return [encode_prepared(v) for v in value]
    if isinstance(value, dict):
        return {k: encode_prepared(v) for k, v in value.items()}
    return value


def decode_prepared(value):
    # Fixed allowlist: never import or instantiate a type named by a document.
    from src.ingestion.parser import Utterance, FigurePlacement, DocumentBlock
    from src.ingestion.pdf_extract import ExtractedPdf, ProseBlock
    from src.ingestion.tabular import SheetGrid
    from src.ingestion.figure_extract import FigureRecord, OfficeFigureResult
    types = {t.__name__: t for t in (
        PreparedDocument, ParsedDocument, Utterance, FigurePlacement,
        DocumentBlock, ExtractedPdf, ProseBlock, SheetGrid, FigureRecord, OfficeFigureResult,
    )}

    def decode(item):
        if isinstance(item, list):
            return [decode(v) for v in item]
        if isinstance(item, dict):
            if set(item) == {"$type", "fields"}:
                if item["$type"] == "tuple":
                    return tuple(decode(v) for v in item["fields"])
                cls = types[item["$type"]]
                return cls(**{k: decode(v) for k, v in item["fields"].items()})
            return {k: decode(v) for k, v in item.items()}
        return item

    result = decode(value)
    if not isinstance(result, PreparedDocument):
        raise ValueError("Worker did not return a prepared document")
    return result
