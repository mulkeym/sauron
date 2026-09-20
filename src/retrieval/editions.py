"""Edition selection over an already-authorized catalog snapshot only."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
import re

from src.ingestion.document_identity import normalize, version_key, VERSION


@dataclass
class EditionSelection:
    doc_ids: list[str]
    decisions: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    missing_details: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)


def facts(doc):
    return (getattr(doc, 'metadata_tags', None) or {}).get('document_identity', {})


def family_relationship(a, b):
    """Return confidence and reason. Similar topics never establish a family."""
    x, y = facts(a), facts(b)
    if getattr(a, 'content_hash', '') and a.content_hash == getattr(b, 'content_hash', ''):
        return 1.0, 'identical original bytes'
    if not x or not y:
        return 0, ''
    if x.get('product_normalized') and y.get('product_normalized') and x['product_normalized'] != y['product_normalized']:
        return 0, 'different products'
    if x.get('identifier') and x.get('identifier') == y.get('identifier'):
        return .99, 'same explicit document identifier'
    if x.get('identifier') and y.get('identifier') and x['identifier'] != y['identifier']:
        for left, right in ((x,y),(y,x)):
            if any(re.search(r'(?i)\b'+re.escape(right['identifier'])+r'\b', s) for s in left.get('supersedes', [])):
                return .98, 'explicit document supersession reference'
        return 0, 'different explicit document identifiers'
    headings_x, headings_y = set(x.get('headings', [])), set(y.get('headings', []))
    overlap = len(headings_x & headings_y) / max(1,len(headings_x | headings_y))
    title_same = x.get('normalized_title') and x.get('normalized_title') == y.get('normalized_title')
    if (title_same and x.get('title_source') != 'filename' and y.get('title_source') != 'filename'
            and x.get('product_normalized') and x.get('product_normalized') == y.get('product_normalized')
            and len(headings_x & headings_y) >= 3 and overlap >= .7):
        return .93, 'same source title/product and matching section structure'
    similarity = SequenceMatcher(None,x.get('normalized_title',''),y.get('normalized_title','')).ratio()
    if title_same or similarity >= .8 and overlap >= .4:
        return .65, 'possible edition; title/structure similarity is not authoritative'
    return 0, ''


def newer(a, b):
    """Strong partial order, or None. Conflicting order signals never pick a winner."""
    x,y = facts(a),facts(b)
    if x.get('conflicts') or y.get('conflicts'):
        return None
    xv,yv = version_key(x.get('revision')),version_key(y.get('revision'))
    xd,yd = x.get('effective_date') or x.get('publication_date'), y.get('effective_date') or y.get('publication_date')
    signals=[]
    if xv and yv and xv != yv:
        signals.append(1 if xv > yv else -1)
    if xd and yd and xd != yd:
        signals.append(1 if xd > yd else -1)
    # A supersession statement naming the other revision or distinct document ID.
    for left,right,direction in ((x,y,1),(y,x,-1)):
        for statement in left.get('supersedes', []):
            id_match = right.get('identifier') and re.search(r'\b'+re.escape(right['identifier'])+r'\b',statement,re.I)
            revision = re.search(r'(?i)\b(?:rev(?:ision)?|edition)\s*[:#-]?\s*('+VERSION+r')\b',statement)
            if id_match and ((revision and revision[1] == right.get('revision')) or left.get('identifier') != right.get('identifier')):
                signals.append(direction)
    return signals[0] if signals and len(set(signals)) == 1 else None


def request_context(question, conversation=None):
    """Last explicit user values win; previous assistant guesses are not evidence."""
    user_turns=[m.get('content','') for m in (conversation or [])[-8:] if m.get('role')=='user' and isinstance(m.get('content'),str)]
    text='\n'.join(t[:4000] for t in user_turns)+ '\n'+question
    versions=re.findall(r'(?i)\b(?:firmware|software|release|version|running)\s*[:=]?\s*v?('+VERSION+r')\b',text)
    environments=re.findall(r'(?i)\b(production|prod|lab|staging|test)\b',text)
    products=re.findall(r'(?im)\b(?:product|platform)\s*[:=]\s*([^\n,;]+)',text)
    return {'software_version': versions[-1] if versions else '',
            'environment': normalize(environments[-1]) if environments else '',
            'product': normalize(products[-1]) if products else '', 'text':text[-12000:]}


def applicable(identity, context):
    """True/False/None: explicit match, explicit exclusion, or unknown."""
    product=identity.get('product_normalized','')
    if product and context.get('product') and product != context['product']:
        return False
    env=normalize(identity.get('environment','')).replace('prod','production') if identity.get('environment')=='prod' else normalize(identity.get('environment',''))
    requested=context.get('environment','')
    requested='production' if requested=='prod' else requested
    if env and requested and env != requested:
        return False
    declarations=identity.get('applicability', [])
    version=context.get('software_version')
    if not declarations or not version:
        return None
    query=version_key(version)
    supported=[]
    for declaration in declarations:
        ranges=re.findall(r'('+VERSION+r')\s*(?:-|–|through|to)\s*('+VERSION+r')',declaration,re.I)
        if ranges:
            supported.append(any(version_key(lo)<=query<=version_key(hi) for lo,hi in ranges))
            continue
        ge=re.search(r'(?:>=|at least)\s*('+VERSION+r')',declaration,re.I)
        if ge:
            supported.append(query>=version_key(ge[1]));continue
        versions=re.findall(r'(?<![\w.])\d+\.\d+(?:\.\d+){0,2}[a-z]?(?![\w.])',declaration,re.I)
        if versions:
            supported.append(any(version==v or version.startswith(v+'.') for v in versions))
    return any(supported) if supported else None


def select_editions(documents, question='', conversation=None):
    documents=sorted(documents,key=lambda d:d.doc_id)
    result=EditionSelection([d.doc_id for d in documents])
    context=request_context(question,conversation)
    historical=bool(re.search(r'(?i)\b(compare|comparison|historical|previous edition|older edition|revision\s+\d|edition\s+\d|as of)\b',question))
    comparison = bool(re.search(r'(?i)\b(compare|comparison|between|versus|vs\.?)\b', question))
    requested_revision = re.search(r'(?i)\b(?:revision|edition)\s+('+VERSION+r')\b', question)
    as_of = re.search(r'(?i)\bas of\s+(\d{4}-\d{2}-\d{2})\b', question)
    groups=[]
    for doc in documents:
        match=None
        for group in groups:
            # Complete linkage prevents a weak/transitive bridge from merging manuals.
            if all(family_relationship(doc,other)[0]>=.9 for other in group):
                match=group;break
        if match is None:groups.append([doc])
        else:match.append(doc)
    for i,doc in enumerate(documents):
        for other in documents[i+1:]:
            confidence,reason=family_relationship(doc,other)
            if .5 <= confidence < .9:
                result.candidates.append({'doc_ids':[doc.doc_id,other.doc_id],'confidence':confidence,'reason':reason})
    selected=[]
    for group in groups:
        family_label=facts(group[0]).get('identifier') or facts(group[0]).get('title') or getattr(group[0], 'filename', group[0].doc_id)
        relevant=(len(groups)==1 or (bool(normalize(family_label)) and normalize(family_label) in normalize(context['text']))
            or any(set(normalize(context['text']).split()) & (set(facts(d).get('product_normalized','').split()) | (set(facts(d).get('normalized_title','').split())-{'guide','manual','user','configuration','deployment','the','and'})) for d in group))
        eligible=[]
        for doc in group:
            identity=facts(doc)
            match=applicable(identity,context)
            future=identity.get('effective_date') and identity['effective_date']>date.today().isoformat()
            if historical or (match is not False and not future):
                if requested_revision and not comparison and identity.get('revision') != requested_revision[1]:
                    continue
                when = identity.get('effective_date') or identity.get('publication_date')
                if as_of and when and when > as_of[1]:
                    continue
                eligible.append(doc)
        if (not historical or as_of) and len(eligible)>1:
            signatures={(tuple(facts(d).get('applicability',[])),normalize(facts(d).get('environment',''))) for d in eligible}
            if len(signatures)>1 and relevant:
                if not context['software_version'] and len({s[0] for s in signatures})>1:
                    result.missing_details.append('software_version')
                if not context['environment'] and len({s[1] for s in signatures})>1:
                    result.missing_details.append('environment')
            winners=[]
            for doc in eligible:
                suppressed=False
                for other in eligible:
                    if other is doc:continue
                    if getattr(doc, 'content_hash', '') and doc.content_hash==getattr(other, 'content_hash', '') and other.doc_id<doc.doc_id:
                        suppressed=True;break
                    # Do not promote a new edition across differing/unknown applicability.
                    same=(facts(doc).get('applicability',[])==facts(other).get('applicability',[]) and facts(doc).get('environment','')==facts(other).get('environment',''))
                    both=(applicable(facts(doc),context) is True and applicable(facts(other),context) is True
                          and facts(doc).get('environment','') == facts(other).get('environment',''))
                    if (same or both) and newer(other,doc)==1:
                        suppressed=True;break
                if not suppressed:winners.append(doc)
            eligible=winners
            if len(eligible)>1 and relevant:
                result.warnings.append(f'Edition/applicability ambiguity in {family_label}; available revisions remain separate evidence.')
        if not eligible and relevant:
            result.warnings.append(f'No edition of {family_label} is confirmed applicable to the requested context.')
        selected.extend(d.doc_id for d in eligible)
        for doc in eligible:
            f=facts(doc)
            result.decisions[doc.doc_id]={'family':family_label,'revision':f.get('revision',''),
                'publication_date':f.get('publication_date'),'effective_date':f.get('effective_date'),
                'applicability':f.get('applicability',[]),'confidence':f.get('confidence',{}),
                'reason':'historical/comparison access' if historical else 'newest established applicable edition; ambiguous editions retained',
                'source_evidence':f.get('source_evidence',[]), 'source_revision':getattr(doc, 'content_hash', ''),
                'family_confidence':min((family_relationship(doc,other)[0] for other in group if other is not doc), default=f.get('confidence',{}).get('family',0))}
    result.doc_ids=sorted(selected)
    selected_records=[d for d in documents if d.doc_id in set(selected)]
    for source in selected_records:
        links=[]
        for target in selected_records:
            if source is target or getattr(target,'doc_type','')!='vsdx':continue
            identifiers=[getattr(target,'filename',''), facts(target).get('identifier','')]
            for reference in facts(source).get('references',[]):
                if any(identifier and re.search(r'(?<![\w-])'+re.escape(identifier)+r'(?![\w-])',reference,re.I) for identifier in identifiers):
                    requested_revision=re.search(r'(?i)\brev(?:ision)?\s*[:#-]?\s*('+VERSION+r')',reference)
                    if requested_revision and requested_revision[1] != facts(target).get('revision'):continue
                    links.append({'doc_id':target.doc_id,'confidence':.99,'reason':'explicit source reference','quote':reference})
        result.decisions[source.doc_id]['diagram_links']=links
    for candidate in result.candidates:
        related=[d for d in documents if d.doc_id in candidate['doc_ids']]
        if any(facts(d).get('normalized_title') and facts(d)['normalized_title'] in normalize(context['text']) for d in related):
            result.warnings.append('Possible related editions: '+', '.join(getattr(d,'filename',d.doc_id) for d in related)+'. Family membership is unconfirmed; neither supersedes the other.')
    result.missing_details=list(dict.fromkeys(result.missing_details))
    return result
