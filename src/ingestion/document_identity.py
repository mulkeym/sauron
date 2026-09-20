"""Conservative, source-grounded edition facts. Never use upload/file timestamps."""
from __future__ import annotations
import hashlib
import re
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 1
VERSION = r'\d+(?:\.\d+){0,3}(?:[a-z])?'


def normalize(value):
    value = re.sub(r'\b(?:rev(?:ision)?|edition|version)\s*[:#-]?\s*'+VERSION, '', value, flags=re.I)
    return ' '.join(re.findall(r'[a-z0-9]+', value.lower()))


def version_key(value):
    if not re.fullmatch(VERSION, value or '', re.I):
        return None
    m = re.fullmatch(r'(\d+(?:\.\d+)*)([a-z]?)', value.lower())
    numbers = [int(v) for v in m[1].split('.')]
    return tuple((numbers + [0] * 4)[:4]) + (m[2],)


def source_date(value):
    value = value.strip()
    # Ambiguous slash dates are evidence, not authority for selecting an edition.
    for fmt in ('%Y-%m-%d', '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y'):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def analyze_document(parsed):
    text = parsed.text or ''
    evidence, facts, conflicts = [], {}, []
    rules = {
        'identifier': r'(?:Document\s+(?:ID|number|no\.?|identifier)|Doc\.?\s+ID)',
        'title': r'(?:Document\s+title|Title)',
        'product': r'(?:Product|Platform)',
        'revision': r'(?:Document\s+)?(?:Revision|Rev\.?|Edition)',
        'publication_date': r'(?:Publication\s+date|Published(?:\s+on)?|Issue\s+date)',
        'effective_date': r'(?:Effective(?:\s+date|\s+from)?)',
        'applicability': r'(?:Applies\s+to|Applicable\s+(?:software|firmware|versions?)|Software(?:\s+versions?)?|Firmware(?:\s+versions?)?)',
        'environment': r'(?:Environment|Deployment\s+environment)',
        'supersedes': r'(?:Supersedes|Replaces)',
        'references': r'(?:Related\s+(?:documents?|manuals?|diagrams?)|References?)',
    }
    # Front matter plus explicit labeled facts throughout; retained evidence is bounded.
    for key, label in rules.items():
        hits = []
        for m in re.finditer(r'(?im)^\s*(?:#+\s*)?'+label+r'\s*[:=]\s*([^\n]+)', text[:500000]):
            value = m[1].strip().strip('|').strip()[:500]
            hits.append(value)
            evidence.append({'field':key,'value':value,'quote':m[0].strip()[:700],
                             'start_char':m.start(),'end_char':m.end()})
            if len(hits) >= 10:
                break
        values = list(dict.fromkeys(hits))
        if key in ('references','supersedes','applicability'):
            facts[key] = values
        elif len(values) == 1:
            facts[key] = values[0]
        elif values:
            conflicts.append(key)
    headings = list(dict.fromkeys(' > '.join(b.section_path) or b.text for b in parsed.blocks
        if b.block_type == 'heading'))[:100]
    if not headings:
        headings = re.findall(r'(?m)^#{1,6}\s+(.+)$', text)[:100]
    title = facts.get('title', '')
    title_source = 'explicit' if title else ''
    if not title and headings:
        title, title_source = headings[0], 'heading'
    if not title:
        title, title_source = Path(parsed.filename).stem, 'filename'
    facts['title'] = title
    facts['title_source'] = title_source
    facts['normalized_title'] = normalize(title)
    facts['headings'] = [normalize(h) for h in headings if len(normalize(h)) >= 3]
    facts['product_normalized'] = normalize(facts.get('product', ''))
    if facts.get('identifier'):
        facts['identifier'] = facts['identifier'].strip().upper()
    raw_revision = facts.get('revision', '')
    if not version_key(raw_revision):
        facts.pop('revision', None)
        if raw_revision:
            conflicts.append('revision')
    for field in ('effective_date', 'publication_date'):
        if field in facts:
            raw = facts[field]
            facts[field] = source_date(raw)
            if not facts[field]:
                conflicts.append(field)
    # Keep explicit history text as evidence, without assuming its last row is current.
    history = re.search(r'(?is)(?:revision|version)\s+history[^\n]*\n(.{0,3000})', text)
    facts['revision_history'] = history[0] if history else ''
    facts['source_evidence'] = evidence
    facts['conflicts'] = sorted(set(conflicts))
    facts['confidence'] = {
        'family': .99 if facts.get('identifier') else (.8 if facts.get('product') and title_source != 'filename' else .2),
        'revision_order': .95 if facts.get('revision') or facts.get('effective_date') or facts.get('publication_date') else 0,
        'applicability': .95 if facts.get('applicability') else 0,
    }
    facts['schema_version'] = SCHEMA_VERSION
    facts['normalized_text_hash'] = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
    return facts
