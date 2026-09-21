#!/usr/bin/env python3
"""Match existing catalog documents to accessible originals and rebuild derived passages."""
import argparse
import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


async def main(args):
    from src.db.metadata import MetadataStore
    from src.retrieval.vector_store import VectorStore
    from src.ingestion.technical_reprocess import reprocess_directory
    metadata=MetadataStore()
    await metadata.init()
    try:
        report=await reprocess_directory(args.source_dir,metadata,VectorStore(),dry_run=not args.apply)
        print(json.dumps(report,indent=2))
        return 1 if any(r.get('error') for r in report['results']) else 0
    finally:await metadata.engine.dispose()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',required=True,type=Path)
    parser.add_argument('--apply',action='store_true',help='Publish reprocessed passages; default only reports matching original hashes')
    raise SystemExit(asyncio.run(main(parser.parse_args())))
