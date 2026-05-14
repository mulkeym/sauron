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
    elif suffix in (".conf", ".cfg", ".ini", ".yaml", ".yml", ".json",
                     ".xml", ".log", ".sh", ".bat", ".ps1", ".py",
                     ".html", ".htm", ".rtf", ".tsv"):
        return _parse_plaintext(path)
    else:
        # Try as plain text for any unrecognized extension
        return _parse_plaintext(path)


def _parse_plaintext(path: Path) -> ParsedDocument:
    """Parse any plain text file (config, log, script, etc.)."""
    text = path.read_text(encoding="utf-8", errors="replace")
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
