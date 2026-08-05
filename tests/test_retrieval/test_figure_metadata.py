from __future__ import annotations

import lancedb
import pyarrow as pa

from src.config import settings
from src.retrieval.models import ChunkMetadata
from src.retrieval.vector_store import VectorStore


def _old_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 3)),
        pa.field("doc_id", pa.string()),
        pa.field("filename", pa.string()),
        pa.field("doc_type", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("start_char", pa.int32()),
        pa.field("acl_groups", pa.list_(pa.string())),
        pa.field("category", pa.string()),
        pa.field("chunk_size_tier", pa.string()),
        pa.field("page", pa.int32()),
        pa.field("speaker", pa.string()),
        pa.field("utterance_type", pa.string()),
    ])


def test_existing_vector_table_migrates_and_round_trips_figure_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "lancedb_path", str(tmp_path / "lance"))
    monkeypatch.setattr(settings, "lancedb_table_name", "chunks")
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    db = lancedb.connect(settings.lancedb_path)
    db.create_table("chunks", schema=_old_schema())

    store = VectorStore()
    assert "figure_id" in store.table.schema.names
    assert "slide" in store.table.schema.names
    meta = ChunkMetadata(
        doc_id="doc-1", filename="manual.pptx", doc_type="pptx",
        chunk_index=4, start_char=0, acl_groups=["ALL"],
        chunk_size_tier="medium", content_type="figure",
        figure_id="s3-fig-004", figure_kind="network", body_index=12,
        section_title="Control Plane", caption="Controller topology",
        source_locator="Figure s3-fig-004, slide 3, Control Plane", slide=3,
    )
    store.upsert(["CTRL-01 connects to CTRL-02"], [[0.1, 0.2, 0.3]], [meta])

    chunks = store.get_chunks_by_doc("doc-1", tier="medium")
    assert len(chunks) == 1
    assert chunks[0].metadata.content_type == "figure"
    assert chunks[0].metadata.figure_id == "s3-fig-004"
    assert chunks[0].metadata.slide == 3
    assert chunks[0].metadata.section_title == "Control Plane"
    assert chunks[0].metadata.caption == "Controller topology"
