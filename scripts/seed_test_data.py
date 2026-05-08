"""Seed the system with test documents for development/demo purposes."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.metadata import MetadataStore
from src.ingestion.pipeline import ingest_document
from src.retrieval.vector_store import VectorStore

FIXTURES = Path(__file__).parent.parent / "test_fixtures"

DOCUMENTS = [
    {"path": FIXTURES / "sample.pdf", "acl_groups": ["finance", "executives"], "category": "finance_policies"},
    {"path": FIXTURES / "sample.docx", "acl_groups": ["it_support", "devops"], "category": "it_runbooks"},
    {"path": FIXTURES / "sample.xlsx", "acl_groups": ["finance", "executives"], "category": "financial_data"},
    {"path": FIXTURES / "sample_transcript.txt", "acl_groups": ["engineering"], "category": "meeting_notes"},
]

async def main():
    vector_store = VectorStore()
    metadata_store = MetadataStore()
    await metadata_store.init()
    for doc_info in DOCUMENTS:
        path = doc_info["path"]
        if not path.exists():
            print(f"  SKIP: {path.name} (file not found)")
            continue
        print(f"  Ingesting: {path.name} -> {doc_info['category']}")
        result = await ingest_document(file_path=path, acl_groups=doc_info["acl_groups"], uploaded_by="seed-script", vector_store=vector_store, metadata_store=metadata_store, category=doc_info["category"])
        print(f"    doc_id={result.doc_id}, chunks={result.chunk_count}")
    print("\nSeeding complete.")

if __name__ == "__main__":
    asyncio.run(main())
