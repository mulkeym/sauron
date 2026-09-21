#!/usr/bin/env python3
"""Retain a locally available original only when it matches the indexed SHA-256.

Run from the Sauron repository: python -m scripts.backfill_original --doc-id ID --file PATH
This is an administrator CLI, not an arbitrary-path HTTP endpoint.
"""

import argparse
import asyncio
from pathlib import Path

from src.config import settings
from src.db.metadata import MetadataStore
from src.sources.storage import OriginalStore


async def backfill(doc_id, source):
    if not settings.source_originals_dir:
        raise ValueError(
            "Set SOURCE_ORIGINALS_DIR to a private persistent directory first"
        )
    metadata = MetadataStore(settings.database_url)
    try:
        doc = await metadata.get_document(doc_id)
        if doc is None:
            raise ValueError("Document not found")
        if not OriginalStore().retain(
            source, doc.doc_id, doc.content_hash, doc.filename
        ):
            raise ValueError(
                "This document type does not have an original-file provider"
            )
        print(
            f"Retained verified original for {doc.doc_id} at revision {doc.content_hash}"
        )
    finally:
        await metadata.engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--file", required=True, type=Path)
    args = parser.parse_args()
    try:
        asyncio.run(backfill(args.doc_id, args.file))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Backfill failed: {exc}\n")


if __name__ == "__main__":
    main()
