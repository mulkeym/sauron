#!/usr/bin/env python3
"""Deterministic synthetic baseline/selector comparison; no LLM quality claims."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ingestion.document_identity import analyze_document
from src.ingestion.parser import ParsedDocument
from src.retrieval.editions import select_editions
from src.agent.strategies.technical import technical_intent

CASES_PATH=Path(__file__).resolve().parents[1]/'tests/fixtures/technical_eval.json'


def make_document(record):
    labels={'identifier':'Document ID','title':'Title','product':'Product','revision':'Revision',
            'publication_date':'Publication date','effective_date':'Effective date',
            'applicability':'Applies to','environment':'Environment','supersedes':'Supersedes','references':'References'}
    body='\n'.join(f'{label}: {record[key]}' for key,label in labels.items() if record.get(key))
    body+='\n\n# Branch Deployment Guide\n## Prerequisites\nAccess required.\n## Configuration steps\n1. Configure the WAN.\n## Verification\nVerify tunnel health.\n'
    parsed=ParsedDocument(record['id']+'.pdf','pdf',body)
    return SimpleNamespace(doc_id=record['id'],filename=record.get('filename',parsed.filename),doc_type=record.get('doc_type','pdf'),
        acl_groups=record.get('acl',['network']),dataset_id=record.get('dataset',1),
        content_hash=record.get('hash',hashlib.sha256((record['id']+body).encode()).hexdigest()),
        metadata_tags={'document_identity':analyze_document(parsed)},created_at=record.get('uploaded',''))


def evaluate_case(case):
    start=time.perf_counter()
    if case['kind']=='intent':
        actual=technical_intent(case['question'])
        return {'case':case['name'],'kind':'intent','passed':actual==case['expected_intent'],'actual':actual,
                'latency_ms':round((time.perf_counter()-start)*1000,3)}
    docs=[make_document(d) for d in case['documents']]
    # The selector deliberately accepts an already-authorized catalog, like QueryScope.
    docs=[d for d in docs if 'network' in d.acl_groups and d.dataset_id==1]
    selection=select_editions(docs,case['question'],case.get('conversation'))
    expected=sorted(case['expected_docs'])
    return {'case':case['name'],'kind':'editions','passed':selection.doc_ids==expected and sorted(selection.missing_details)==sorted(case['expected_missing']),
            'baseline_correct':sorted(d.doc_id for d in docs)==expected and not case['expected_missing'],
            'selected':selection.doc_ids,'expected':expected,'missing_details':selection.missing_details,
            'warnings':selection.warnings,'latency_ms':round((time.perf_counter()-start)*1000,3)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    cases=json.loads(CASES_PATH.read_text())['cases']
    results=[evaluate_case(case) for case in cases]
    editions=[r for r in results if r['kind']=='editions']
    report={'synthetic':True,'cases':len(results),'passed':sum(r['passed'] for r in results),
            'baseline_edition_correct':sum(r['baseline_correct'] for r in editions),
            'selected_edition_correct':sum(r['passed'] for r in editions),
            'edition_cases':len(editions),'llm_answer_quality':None,
            'limitations':'Source fixtures and deterministic selectors/routing only. No protected corpus, real embedding ranking or generated-answer entailment evaluation.',
            'results':results}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    for r in results:
        if not r['passed']:print('FAILED',json.dumps(r))
    return 0 if report['passed']==len(results) else 1


if __name__=='__main__':raise SystemExit(main())
