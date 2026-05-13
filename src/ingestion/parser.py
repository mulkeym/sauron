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
    elif suffix in (".xlsx", ".csv"):
        return _parse_spreadsheet(path)
    elif suffix == ".md":
        return _parse_markdown(path)
    elif suffix == ".txt":
        return _parse_transcript(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _parse_markdown(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8")
    text = _strip_web_boilerplate(text)
    return ParsedDocument(filename=path.name, doc_type="markdown", text=text)


def _strip_web_boilerplate(text: str) -> str:
    """Strip website navigation, headers, footers, and social media links from markdown.

    Many documents are scraped from government websites and contain huge
    amounts of navigation boilerplate that pollutes chunking and entity extraction.
    """
    import re

    lines = text.split("\n")
    content_lines = []
    in_content = False
    consecutive_link_lines = 0

    for line in lines:
        stripped = line.strip()

        # Detect main content start: first heading (# ...) or bold section header (**ARMY**)
        if not in_content:
            if re.match(r'^#{1,3}\s+\w', stripped) and 'skip to' not in stripped.lower():
                in_content = True
            elif re.match(r'^\*\*[A-Z][A-Z\s]+\*\*$', stripped):  # **ARMY**, **NAVY**
                in_content = True

        if not in_content:
            continue

        # Skip lines that are purely links/navigation
        is_link_line = bool(re.match(r'^\s*\*?\s*\[', stripped)) and '](' in stripped and len(stripped.split('](')) > 1
        is_social = any(s in stripped.lower() for s in ['facebook', 'instagram', 'linkedin', 'youtube', 'twitter', '/#x)', '/#facebook'])
        is_nav = stripped.startswith('*   [') and stripped.count('](') >= 1 and len(stripped) < 200

        if is_social:
            continue
        if is_nav or is_link_line:
            consecutive_link_lines += 1
            if consecutive_link_lines >= 3:
                continue  # skip runs of navigation links
        else:
            consecutive_link_lines = 0

        # Skip footer patterns
        if any(p in stripped.lower() for p in [
            'privacy & security', 'links disclaimer', 'no fear act',
            'information quality', 'plain writing act', 'usa.gov',
            'hosted by department', 'web.mil', 'addtoany', 'thanks for sharing',
            'small business act', 'foia', 'accessibility/508',
        ]):
            continue

        # Skip sharing widgets
        if stripped in ('×', 'Share', '**Copy Link**'):
            continue

        content_lines.append(line)

    result = "\n".join(content_lines).strip()

    # If stripping removed too much, return original
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


def _parse_docx(path: Path) -> ParsedDocument:
    doc = DocxDocument(str(path))
    parts = []
    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            parts.append(f"\n## {para.text}\n")
        elif para.text.strip():
            parts.append(para.text)
    text = "\n".join(parts)
    return ParsedDocument(filename=path.name, doc_type="docx", text=text)


def _parse_spreadsheet(path: Path) -> ParsedDocument:
    wb = load_workbook(str(path), read_only=True)
    parts = []
    for sheet_name in wb.sheetnames:
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
    return ParsedDocument(filename=path.name, doc_type="xlsx", text=text, metadata={"sheet_names": wb.sheetnames})


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
