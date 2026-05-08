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
