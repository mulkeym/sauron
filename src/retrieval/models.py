from __future__ import annotations
from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    doc_id: str
    filename: str
    doc_type: str
    chunk_index: int
    start_char: int
    acl_groups: list[str]
    category: str = ""
    chunk_size_tier: str = "medium"  # "small" (512), "medium" (1024), "large" (2048)
    page: int | None = None
    speaker: str | None = None
    utterance_type: str | None = None
    content_type: str = "text"  # "text" | "figure" | "table"
    figure_id: str | None = None
    figure_kind: str | None = None
    body_index: int | None = None
    section_title: str | None = None
    caption: str | None = None
    source_locator: str | None = None
    slide: int | None = None


class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: ChunkMetadata


class Citation(BaseModel):
    doc_id: str
    filename: str
    doc_type: str
    chunk_index: int
    page: int | None = None
    snippet: str
    relevance: float
    source_url: str = ""
    figure_id: str | None = None
    section_title: str | None = None
    caption: str | None = None
    slide: int | None = None
