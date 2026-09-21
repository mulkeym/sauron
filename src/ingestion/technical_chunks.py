"""Bounded section-aware chunks. Preserve source lines and block ordering."""
from dataclasses import dataclass
import hashlib
import re
from src.ingestion.chunker import Chunk

ROLE_WORDS = {
    'prerequisites': r'prerequisites?|before you (?:begin|start)|requirements?',
    'steps': r'procedure|configuration steps|configure|deployment steps|installation|^\s*\d+[.)]\s',
    'warnings': r'warning|caution|important|disrupt|outage|service impact',
    'verification': r'verif(?:y|ication)|validation|confirm|expected (?:output|result)',
    'rollback': r'rollback|roll back|revert|restore previous',
    'symptoms': r'symptoms?|observed|error|failure|packet loss|timeout',
    'causes': r'possible causes?|potential causes?|cause of|due to',
    'diagnostics': r'diagnostic|check|test|inspect|show\s',
    'interpretation': r'if .* (?:then|indicates|means)|indicates|expected|otherwise',
    'correction': r'remediat|corrective|resolution|workaround|repair',
}


@dataclass
class TechnicalChunk(Chunk):
    section_id: str = ''
    section_path: str = ''
    evidence_role: str = ''
    page: int | None = None
    body_index: int | None = None
    source_locator: str = ''


def roles(text):
    return [name for name,pattern in ROLE_WORDS.items() if re.search(pattern,text,re.I|re.M)]


def _pieces(text, limit):
    """Split at original line boundaries, never rewrite spaces or command tokens."""
    current=''
    for line in text.splitlines(keepends=True):
        if current and len(current)+len(line)>limit:
            yield current; current=''
        if len(line)>limit:
            if current:yield current;current=''
            # Oversize source lines are retained atomically up to the hard section cap.
            from src.config import settings
            for start in range(0,len(line),settings.technical_section_max_chars):
                yield line[start:start+settings.technical_section_max_chars]
        else:current+=line
    if current:yield current


def build_technical_chunks(prepared, chunk_size):
    parsed=prepared.parsed
    units=[]
    if parsed.doc_type=='pdf' and prepared.pdf is not None:
        for i,b in enumerate(prepared.pdf.prose_blocks):
            if b.content_type!='figure':units.append((b.text,[],b.page+1,i,'paragraph'))
        for grid in prepared.pdf.table_grids:
            match=re.match(r'p(\d+)',grid.sheet_name)
            page=int(match[1])+1 if match else None
            rows=[' | '.join('' if c is None else str(c) for c in row) for row in grid.rows]
            if rows:
                # Repeat the first source row for each bounded table fragment.
                for j in range(1,max(2,len(rows)),max(1,chunk_size//200)):
                    body='\n'.join([rows[0]]+rows[j:j+max(1,chunk_size//200)])
                    units.append((body,[grid.sheet_name],page,len(units),'table'))
        units.sort(key=lambda u:((u[2] or 0),u[3]))
    elif parsed.blocks:
        units=[(b.text,b.section_path,b.page+1 if b.page is not None else None,b.body_index,b.block_type)
               for b in parsed.blocks if b.block_type!='figure' and b.text.strip()]
    else:
        units=[(parsed.text,[],None,0,'paragraph')]
    chunks=[];offset=0;path=[];section_serial=0
    for raw, declared, page, body, kind in units:
        if declared:
            if list(declared)!=path:section_serial+=1
            path=list(declared)
        if kind=='heading':
            if not declared:path=[raw.strip()];section_serial+=1
            continue
        # PDF/plaintext headings require explicit numbering/Markdown or a named
        # procedure role; typography alone is not a reliable semantic boundary.
        blocks=re.split(r'(?m)(?=^#{1,6}\s+|^\d+(?:\.\d+)+\s+[A-Z]|^(?:Prerequisites|Warnings|Verification|Rollback|Procedure|Diagnostics|Possible causes)\s*:?\s*$)',raw)
        for block in blocks:
            if not block.strip():continue
            first=block.splitlines()[0]
            heading=re.match(r'^(#{1,6})\s+(.+)',first)
            numbered=re.match(r'^(\d+(?:\.\d+)+)\s+(.+)',first)
            if heading or numbered:
                level=len(heading[1]) if heading else len(numbered[1].split('.'))
                title=heading[2] if heading else first
                path=path[:level-1]+[title];section_serial+=1
            elif re.fullmatch(r'(Prerequisites|Warnings|Verification|Rollback|Procedure|Diagnostics|Possible causes)\s*:?',first,re.I):
                path=path[:1]+[first];section_serial+=1
            # Containing section includes child headings (e.g. procedure/warnings).
            parent = path[:-1] if len(path)>1 and re.fullmatch(r'Prerequisites|Warnings|Verification|Rollback|Procedure|Steps|Diagnostics|Possible causes', path[-1].strip(': '), re.I) else path
            containing=' > '.join(parent) or 'Document'
            section_id=hashlib.sha256((containing).encode()).hexdigest()[:16]
            for piece in _pieces(block,chunk_size):
                location=(f'page {page}, ' if page else '')+f'body block {body}; '+(' > '.join(path) or 'Document')
                chunks.append(TechnicalChunk(piece,len(chunks),offset,section_id,' > '.join(path),
                    ','.join(roles((' > '.join(path))+'\n'+piece)),page,body,location))
                offset+=len(piece)
    return chunks


def chunk_metadata(chunk):
    if not isinstance(chunk,TechnicalChunk):return {}
    return {'section_id':chunk.section_id,'section_path':chunk.section_path,
            'evidence_role':chunk.evidence_role,'page':chunk.page,'body_index':chunk.body_index,
            'section_title':chunk.section_path.split(' > ')[-1] or None,
            'source_locator':chunk.source_locator}


def index_text(context, chunk):
    if isinstance(chunk, TechnicalChunk):
        return (f"Section: {chunk.section_path}\n\n" if chunk.section_path else '') + chunk.text
    return f"{context}\n\n{chunk.text}"
