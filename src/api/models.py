from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str
    groups: list[str] = []

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    doc_type: str
    chunk_count: int

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    doc_type: str
    category: str
    acl_groups: list[str]
    chunk_count: int

class QueryRequest(BaseModel):
    question: str

class CitationResponse(BaseModel):
    doc_id: str
    filename: str
    doc_type: str
    chunk_index: int
    page: int | None = None
    snippet: str
    relevance: float

class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    cached: bool = False
    cached_query: str | None = None

class AsyncQuerySubmitResponse(BaseModel):
    token: str
    status: str

class AsyncQueryStatusResponse(BaseModel):
    token: str
    status: str
    step: str
    steps: list[dict] = []                 # timeline: [{"step": label, "at": elapsed_s}, ...]
    classification: dict | None = None     # classify detail: query_type, reason, sub_tasks, strategy_memory
    answer: str | None = None
    citations: list[CitationResponse] = []
    cached: bool = False
    cached_query: str | None = None
    error: str | None = None
    created_at: float
    completed_at: float | None = None
