"""Human-readable citation presentation; canonical evidence IDs stay in stored data."""
from __future__ import annotations

from html import escape
import re
from urllib.parse import quote, urlsplit, urlunsplit


def _record(citation):
    return citation.model_dump() if hasattr(citation, "model_dump") else citation


def citation_label(citation) -> str:
    c = _record(citation)
    filename = str(c.get("filename") or "Source")
    parts = []
    if c.get("page") is not None:
        parts.append(f"page {c['page']}")
    if c.get("slide") is not None:
        parts.append(f"slide {c['slide']}")
    section = " ".join(str(c.get("section_title") or "").split())
    if section:
        parts.append(section[:120] + ("…" if len(section) > 120 else ""))
    if c.get("figure_id"):
        parts.append("diagram")
    kind = c.get("source_kind", "document")
    if kind == "query_result":
        parts.append("query result")
    elif kind == "derived":
        parts.append("derived summary")
    elif not parts and c.get("chunk_index") is not None:
        parts.append(f"passage {int(c['chunk_index']) + 1}")
    revision = c.get("edition", {}).get("revision")
    if revision:
        parts.append(f"revision {revision}")
    return " ".join(filename.split()) + (" — " + ", ".join(parts) if parts else "")


def citation_details(citation) -> dict:
    """Keep machine provenance alongside a ready-to-display label for clients."""
    return {**_record(citation), "display_label": citation_label(citation)}


def citation_url(citation) -> str:
    """Use an existing source URL only; never invent public chunk/download routes."""
    c = _record(citation)
    url = c.get("source_url") or ""
    if any(ord(ch) < 32 or ch == "\\" for ch in url):
        return ""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
            return ""
        if parsed.path.lower().endswith('.pdf') and c.get('page') is not None and not parsed.fragment:
            parsed = parsed._replace(fragment=f"page={c['page']}")
        return quote(urlunsplit(parsed), safe=":/?#@!$&'*+,;=%-._~")
    except ValueError:
        return ""


def citation_markdown(citation, *, target=None) -> str:
    # Metadata is untrusted. Escape labels independently from validated URLs.
    label = re.sub(r"([\\`*_{}\[\]!|])", r"\\\1", escape(citation_label(citation), quote=False))
    url = target if target is not None else citation_url(citation)
    return f"[{label}]({url})" if url else f"[{label}]"


# Leave literal code and existing Markdown links untouched. Only exact, supplied
# evidence markers (or legacy ordinal references) may become source labels.
_MARKER = re.compile(
    r"(?P<literal>```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`+[^`]*`+|"
    r"!?\[[^\]\n]*\]\([^\n]*?\)|\[[^\]\n]*\]\[[^\]\n]*\])"
    r"|(?<!\\)\[(?P<id>E[A-Za-z0-9_-]+|\d+)\]"
)


def render_citations(answer, citations, *, local_links=False, aliases=None):
    """Format at delivery time, after validation, including canonical cached answers."""
    if not answer:
        return answer
    labels = {}
    for ordinal, citation in enumerate(citations, 1):
        c = _record(citation)
        key = c.get("evidence_id") or str(ordinal)
        label = citation_markdown(c, target=f"#citation-{ordinal}" if local_links else None)
        labels[key] = label
    if aliases:
        labels.update({alias: labels[eid] for alias, eid in aliases.items() if eid in labels})
    return _MARKER.sub(lambda m: m[0] if m.group('literal') else labels.get(m['id'], m[0]), answer)


class CitationStream:
    """Buffer Markdown lines so partial IDs and literal code render consistently."""
    def __init__(self, citations, *, aliases=None):
        self.citations = citations
        self.aliases = aliases
        self.buffer = ""
        self.fence = None

    def feed(self, text, *, final=False):
        self.buffer += text
        cutoff = len(self.buffer) if final else self.buffer.rfind('\n') + 1
        ready, self.buffer = self.buffer[:cutoff], self.buffer[cutoff:]
        rendered = []
        for line in ready.splitlines(keepends=True):
            marker = re.match(r"^\s*(`{3,}|~{3,})", line)
            if self.fence:
                rendered.append(line)
                if marker and marker[1][0] == self.fence[0] and len(marker[1]) >= len(self.fence):
                    self.fence = None
            elif marker:
                self.fence = marker[1]
                rendered.append(line)
            else:
                rendered.append(render_citations(line, self.citations, aliases=self.aliases))
        return ''.join(rendered)


def map_prose(text, transform, *, code_replacement=None):
    """Transform prose while leaving fenced and inline code byte-for-byte intact."""
    pattern = re.compile(r"(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`+[^`]*`+)")
    return ''.join((part if code_replacement is None else code_replacement) if i % 2 else transform(part)
                   for i, part in enumerate(pattern.split(text)))
