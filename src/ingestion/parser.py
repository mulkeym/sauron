from dataclasses import dataclass, field
from pathlib import Path

from docx import Document as DocxDocument
from openpyxl import load_workbook


@dataclass
class Utterance:
    speaker: str
    text: str
    utterance_type: str  # "question", "statement"


@dataclass
class ParsedDocument:
    filename: str
    doc_type: str  # "pdf", "docx", "xlsx", "transcript"
    text: str
    metadata: dict = field(default_factory=dict)
    utterances: list[Utterance] = field(default_factory=list)


def parse_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix == ".docx":
        return _parse_docx(path)
    elif suffix in (".xlsx", ".xlsm", ".xls", ".csv", ".tsv"):
        return _parse_spreadsheet(path)
    elif suffix == ".md":
        return _parse_markdown(path)
    elif suffix == ".txt":
        return _parse_transcript(path)
    elif suffix in (".conf", ".cfg", ".ini", ".yaml", ".yml", ".json",
                     ".xml", ".log", ".sh", ".bat", ".ps1", ".py",
                     ".html", ".htm", ".rtf"):
        return _parse_plaintext(path)
    else:
        # Try as plain text for any unrecognized extension
        return _parse_plaintext(path)


def _parse_plaintext(path: Path) -> ParsedDocument:
    """Parse any plain text file (config, log, script, etc.)."""
    raw = path.read_bytes()
    # Guard: refuse to ingest binary files as text. Without this, an
    # unhandled binary format (e.g. a legacy .xls routed here) gets decoded
    # into mojibake and silently embedded into the vector store.
    if b"\x00" in raw[:8192]:
        raise ValueError(
            f"{path.name}: appears to be a binary file, not text; refusing to ingest as plaintext"
        )
    text = raw.decode("utf-8", errors="replace")
    doc_type = path.suffix.lstrip(".") or "text"
    return ParsedDocument(filename=path.name, doc_type=doc_type, text=text)


def _parse_markdown(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8")
    text = _strip_web_boilerplate(text)
    return ParsedDocument(filename=path.name, doc_type="markdown", text=text)


def _strip_web_boilerplate(text: str) -> str:
    """Strip web boilerplate from markdown documents.

    Works on any scraped website by removing lines that are structurally
    boilerplate (navigation links, social media, footers) while preserving
    actual content paragraphs. Generic enough for any document type.
    """
    import re

    lines = text.split("\n")
    kept = []

    for line in lines:
        stripped = line.strip()

        # Keep blank lines (preserve paragraph structure)
        if not stripped:
            kept.append(line)
            continue

        # Remove: lines that are purely markdown links with no meaningful text
        # e.g., "*   [News](https://...)" or "[](https://...)"
        plain_text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', stripped)  # extract link text
        plain_text = re.sub(r'[*_\[\]()#|>!]', '', plain_text).strip()  # strip markdown formatting

        link_count = stripped.count('](')

        # Line is a navigation list item: "* [Link](url)" with minimal text
        if link_count >= 1 and len(plain_text) < 30 and stripped.startswith('*'):
            continue

        # Line is mostly links: 2+ links and plain text is minimal
        if link_count >= 2 and len(plain_text) < 20:
            continue

        # Line is an empty link: [](url) or [![Image](url)](url)
        if re.match(r'^\s*\[?\[?\]?\(', stripped) and len(plain_text) < 5:
            continue

        # Remove: social media links
        if any(s in stripped.lower() for s in ['facebook.com', 'instagram.com', 'linkedin.com',
               'youtube.com', 'twitter.com', '/#facebook', '/#x)', '/#email']):
            continue

        # Remove: common website boilerplate phrases
        if any(p in stripped.lower() for p in [
            'skip to main content', 'official websites use .gov',
            'secure .gov websites', 'how you know', 'share sensitive information',
            'official government organization', 'safely connected',
            'addtoany', 'thanks for sharing', 'previous next slideshow',
        ]):
            continue

        # Remove: sharing widgets
        if stripped in ('×', 'Share', '**Copy Link**', '---', 'Search', 'Search Search'):
            continue

        # Remove: image-only lines
        if re.match(r'^!\[.*\]\(.*\)$', stripped) or re.match(r'^\[!\[.*\]\(.*\)\]\(.*\)$', stripped):
            continue

        kept.append(line)

    # Collapse multiple consecutive blank lines
    result = re.sub(r'\n{4,}', '\n\n\n', "\n".join(kept)).strip()

    # Safety: if we removed >90%, something went wrong — keep original
    if len(result) < len(text) * 0.1 and len(text) > 500:
        return text

    return result


def _parse_pdf(path: Path) -> ParsedDocument:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
        text = "\n".join(pages)
        return ParsedDocument(filename=path.name, doc_type="pdf", text=text)
    except Exception:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(str(path))
        text = "\n".join(str(el) for el in elements)
        return ParsedDocument(filename=path.name, doc_type="pdf", text=text)


def _docx_para_text(para) -> str | None:
    """Format one paragraph; None if empty."""
    text = (para.text or "").strip()
    if not text:
        return None
    style = getattr(getattr(para, "style", None), "name", "") or ""
    if style.startswith("Heading"):
        return f"\n## {text}\n"
    return text


def _docx_table_text(table) -> str | None:
    """Flatten a Word table to readable prose (story-book layouts use tables heavily)."""
    rows_out: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        seen: set[str] = set()
        for cell in row.cells:
            # python-docx repeats merged cells — de-dupe consecutive identical text
            ct = "\n".join(
                p.text.strip() for p in cell.paragraphs if p.text and p.text.strip()
            ).strip()
            if not ct or ct in seen:
                continue
            seen.add(ct)
            cells.append(ct)
        if cells:
            rows_out.append(" | ".join(cells) if len(cells) > 1 else cells[0])
    if not rows_out:
        return None
    return "\n".join(rows_out)


def _iter_docx_blocks(doc):
    """Yield paragraphs and tables in document-body order.

    ``Document.paragraphs`` skips table cell text entirely — children's books
    and many designed layouts put the story in tables, so ordered body walk
    is required for correct capture.
    """
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = doc.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _parse_docx(path: Path) -> ParsedDocument:
    doc = DocxDocument(str(path))
    parts: list[str] = []
    seen_norm: set[str] = set()

    def _add(block: str | None) -> None:
        if not block:
            return
        # Collapse whitespace for de-dupe (cover page often repeats title)
        key = " ".join(block.split())
        if key in seen_norm:
            return
        seen_norm.add(key)
        parts.append(block.strip())

    for block in _iter_docx_blocks(doc):
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        if isinstance(block, Paragraph):
            _add(_docx_para_text(block))
        elif isinstance(block, Table):
            _add(_docx_table_text(block))

    # Headers / footers (page numbers etc. — usually small, but keep searchable)
    for section in doc.sections:
        for hf in (section.header, section.footer):
            if hf is None:
                continue
            for para in hf.paragraphs:
                _add(_docx_para_text(para))
            for table in hf.tables:
                _add(_docx_table_text(table))

    text = "\n\n".join(parts)
    return ParsedDocument(filename=path.name, doc_type="docx", text=text)


def _sniff_workbook_format(path: Path) -> str | None:
    """Detect a workbook's real format from its leading magic bytes.

    Extensions lie: some sources (e.g. OPM) serve modern OOXML workbooks under
    a legacy ``.xls`` name. Dispatching on content instead of extension is what
    lets those parse correctly rather than failing or producing mojibake.

    Returns "ooxml" (ZIP-based .xlsx/.xlsm), "ole" (legacy binary .xls), or
    None (not a recognized binary workbook — treat as delimited text).
    """
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:4] == b"PK\x03\x04":
        return "ooxml"
    if head == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "ole"
    return None


def _parse_spreadsheet(path: Path) -> ParsedDocument:
    """Parse a spreadsheet to text, dispatching on actual content.

    openpyxl reads OOXML (.xlsx/.xlsm), xlrd reads legacy OLE .xls, and
    .csv/.tsv are plain delimited text. We sniff magic bytes first so a
    mislabeled file (e.g. OOXML content named .xls) is parsed by the right
    reader instead of being read as binary mojibake.
    """
    fmt = _sniff_workbook_format(path)
    if fmt == "ooxml":
        return _parse_xlsx(path)
    if fmt == "ole":
        return _parse_legacy_xls(path)
    # Not a binary workbook — delimited text (covers .csv/.tsv and any
    # text content mislabeled with a spreadsheet extension).
    return _parse_delimited(path)


def _parse_xlsx(path: Path) -> ParsedDocument:
    import io

    # Read via BytesIO so openpyxl dispatches on content, not the filename —
    # it otherwise refuses any path not ending in .xlsx/.xlsm, which breaks
    # the common case of OOXML content served under a .xls name.
    with open(path, "rb") as fh:
        wb = load_workbook(io.BytesIO(fh.read()), read_only=True)
    sheet_names = list(wb.sheetnames)
    parts = []
    for sheet_name in sheet_names:
        ws = wb[sheet_name]
        parts.append(f"Sheet: {sheet_name}")
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        headers = [str(h) if h else "" for h in rows[0]]
        parts.append(" | ".join(headers))
        for row in rows[1:]:
            parts.append(" | ".join(str(c) if c is not None else "" for c in row))
    wb.close()
    text = "\n".join(parts)
    return ParsedDocument(filename=path.name, doc_type="xlsx", text=text, metadata={"sheet_names": sheet_names})


def _parse_legacy_xls(path: Path) -> ParsedDocument:
    """Parse a legacy binary .xls workbook via xlrd."""
    import xlrd

    book = xlrd.open_workbook(str(path))
    sheet_names = book.sheet_names()
    parts = []
    for sheet in book.sheets():
        parts.append(f"Sheet: {sheet.name}")
        for r in range(sheet.nrows):
            cells = sheet.row_values(r)
            parts.append(" | ".join("" if c is None else str(c) for c in cells))
    text = "\n".join(parts)
    return ParsedDocument(filename=path.name, doc_type="xls", text=text, metadata={"sheet_names": sheet_names})


def _parse_delimited(path: Path) -> ParsedDocument:
    """Parse a .csv/.tsv file into ' | '-joined rows."""
    import csv

    # Guard: a binary file that slipped past the workbook sniff must not be
    # decoded into mojibake and ingested as "delimited text".
    if b"\x00" in path.read_bytes()[:8192]:
        raise ValueError(
            f"{path.name}: appears to be a binary file, not delimited text; refusing to ingest"
        )
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    parts = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f, delimiter=delimiter):
            parts.append(" | ".join(row))
    text = "\n".join(parts)
    return ParsedDocument(filename=path.name, doc_type=path.suffix.lstrip(".") or "csv", text=text)


def _parse_transcript(path: Path) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8")
    lines = raw.strip().split("\n")
    metadata = {}
    utterances = []
    text_parts = []

    for line in lines:
        line = line.strip()
        if not line or line == "---":
            continue
        if ":" in line and not any(line.startswith(f"{name}:") for name in _extract_speaker_names(lines)):
            key, _, value = line.partition(":")
            if key.strip() in ("Meeting", "Date", "Location", "Attendees"):
                metadata[key.strip().lower()] = value.strip()
                text_parts.append(line)
                continue
        if ":" in line:
            speaker, _, text = line.partition(":")
            speaker = speaker.strip()
            text = text.strip()
            is_question = text.rstrip().endswith("?")
            utterances.append(Utterance(speaker=speaker, text=text, utterance_type="question" if is_question else "statement"))
            text_parts.append(line)
        else:
            text_parts.append(line)
    return ParsedDocument(filename=path.name, doc_type="transcript", text="\n".join(text_parts), metadata=metadata, utterances=utterances)


def _extract_speaker_names(lines: list[str]) -> set[str]:
    from collections import Counter
    names = Counter()
    for line in lines:
        if ":" in line:
            name = line.split(":")[0].strip()
            if name and len(name) < 40 and name not in ("Meeting", "Date", "Location", "Attendees"):
                names[name] += 1
    return {name for name, count in names.items() if count >= 1}
