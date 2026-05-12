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
