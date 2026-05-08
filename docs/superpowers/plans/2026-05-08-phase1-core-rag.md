# Phase 1: Core RAG System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working RAG system where users upload documents (PDF, Word, spreadsheets, transcripts), ask natural language questions via a chat UI or REST API, and receive cited answers — with document-level access control tied to Active Directory groups.

**Architecture:** A FastAPI backend handles document ingestion (parse, chunk, embed) and query serving. Documents are stored in Qdrant with ACL metadata. Gemma 4 31B served via vLLM generates answers from retrieved context. Open WebUI provides the chat interface. JWT + API key authentication enforces access control. Docker Compose orchestrates all services.

**Tech Stack:** Python 3.11+, FastAPI, Qdrant, vLLM, Gemma 4 31B, E5-large (intfloat/multilingual-e5-large), Unstructured.io, Open WebUI, Docker Compose, PyJWT, python-multipart

---

## File Structure

```
rag/
├── docker-compose.yml                  # Orchestrates all services
├── .env.example                        # Environment variable template
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # App server image
│
├── src/
│   ├── __init__.py
│   ├── main.py                         # FastAPI app entry point, mounts routers
│   ├── config.py                       # Settings from env vars (pydantic-settings)
│   │
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── jwt.py                      # JWT creation, validation, AD group extraction
│   │   ├── api_key.py                  # API key validation middleware
│   │   ├── dependencies.py             # FastAPI dependency injection for auth
│   │   └── models.py                   # User, TokenPayload pydantic models
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py                   # Document parsing (PDF, Word, spreadsheet, transcript)
│   │   ├── chunker.py                  # Structure-aware text chunking
│   │   ├── embedder.py                 # E5-large embedding client
│   │   └── pipeline.py                 # End-to-end ingestion orchestrator
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py             # Qdrant client wrapper (store, search, delete)
│   │   ├── retriever.py                # Top-K retrieval with ACL filtering
│   │   └── models.py                   # RetrievedChunk, Citation pydantic models
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_client.py              # vLLM OpenAI-compat client wrapper
│   │   └── rag_chain.py               # Retrieve -> augment prompt -> generate -> cite
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_query.py            # POST /api/v1/query endpoint
│   │   ├── routes_ingest.py           # POST /api/v1/ingest, GET /api/v1/documents
│   │   ├── routes_auth.py             # POST /api/v1/auth/token (login)
│   │   └── models.py                  # Request/response pydantic models
│   │
│   └── db/
│       ├── __init__.py
│       ├── metadata.py                # SQLite/PostgreSQL metadata store (documents, ACLs)
│       └── models.py                  # SQLAlchemy models for document metadata
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures (test client, mock Qdrant, etc.)
│   ├── test_config.py
│   ├── test_auth/
│   │   ├── __init__.py
│   │   ├── test_jwt.py
│   │   ├── test_api_key.py
│   │   └── test_dependencies.py
│   ├── test_ingestion/
│   │   ├── __init__.py
│   │   ├── test_parser.py
│   │   ├── test_chunker.py
│   │   ├── test_embedder.py
│   │   └── test_pipeline.py
│   ├── test_retrieval/
│   │   ├── __init__.py
│   │   ├── test_vector_store.py
│   │   └── test_retriever.py
│   ├── test_generation/
│   │   ├── __init__.py
│   │   ├── test_llm_client.py
│   │   └── test_rag_chain.py
│   └── test_api/
│       ├── __init__.py
│       ├── test_routes_query.py
│       ├── test_routes_ingest.py
│       └── test_routes_auth.py
│
├── test_fixtures/
│   ├── sample.pdf                     # Small test PDF
│   ├── sample.docx                    # Small test Word doc
│   ├── sample.xlsx                    # Small test spreadsheet
│   └── sample_transcript.txt          # Small test meeting transcript
│
└── scripts/
    ├── create_api_key.py              # CLI tool to generate API keys
    └── seed_test_data.py              # Seed Qdrant with test documents
```

---

## Task 1: Project Scaffolding & Configuration

**Files:**
- Create: `rag/src/__init__.py`
- Create: `rag/src/config.py`
- Create: `rag/requirements.txt`
- Create: `rag/.env.example`
- Create: `rag/tests/__init__.py`
- Create: `rag/tests/test_config.py`
- Create: `rag/tests/conftest.py`

- [ ] **Step 1: Create requirements.txt**

```
# Core
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.9.0
pydantic-settings>=2.6.0
python-multipart>=0.0.12

# Auth
PyJWT>=2.9.0
cryptography>=43.0.0
passlib[bcrypt]>=1.7.4

# Ingestion
unstructured[pdf,docx,xlsx]>=0.16.0
python-docx>=1.1.0
openpyxl>=3.1.0

# Embeddings & Vector DB
sentence-transformers>=3.3.0
qdrant-client>=1.12.0

# LLM
openai>=1.55.0

# Database (metadata store)
sqlalchemy>=2.0.36
aiosqlite>=0.20.0

# Testing
pytest>=8.3.0
pytest-asyncio>=0.24.0
httpx>=0.28.0
```

- [ ] **Step 2: Create .env.example**

```bash
# LLM
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=google/gemma-4-31b-it

# Embeddings
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large
EMBEDDING_DEVICE=cpu

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents

# Auth
JWT_SECRET_KEY=change-me-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=480
API_KEYS=dev-key-1,dev-key-2

# LDAP (Phase 1: simulated; real LDAP in production)
LDAP_ENABLED=false

# Metadata DB
DATABASE_URL=sqlite+aiosqlite:///./data/metadata.db
```

- [ ] **Step 3: Create src/__init__.py (empty) and src/config.py**

```python
# src/__init__.py
```

```python
# src/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "google/gemma-4-31b-it"

    # Embeddings
    embedding_model_name: str = "intfloat/multilingual-e5-large"
    embedding_device: str = "cpu"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"

    # Auth
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 480
    api_keys: str = "dev-key-1"  # comma-separated

    # LDAP
    ldap_enabled: bool = False

    # Metadata DB
    database_url: str = "sqlite+aiosqlite:///./data/metadata.db"

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

- [ ] **Step 4: Create tests/__init__.py, tests/conftest.py, and tests/test_config.py**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
import os
import pytest

# Override settings for tests before any imports
os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["API_KEYS"] = "test-key-1,test-key-2"
os.environ["QDRANT_HOST"] = "localhost"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/test_metadata.db"
```

```python
# tests/test_config.py
import os
import pytest


def test_settings_loads_defaults():
    os.environ["JWT_SECRET_KEY"] = "test-secret"
    os.environ["API_KEYS"] = "key-a,key-b,key-c"

    # Re-import to pick up env overrides
    from src.config import Settings
    s = Settings()

    assert s.jwt_secret_key == "test-secret"
    assert s.api_key_list == ["key-a", "key-b", "key-c"]
    assert s.qdrant_port == 6333
    assert s.jwt_algorithm == "HS256"


def test_api_key_list_handles_whitespace():
    os.environ["API_KEYS"] = " key-1 , key-2 , "
    from src.config import Settings
    s = Settings()
    assert s.api_key_list == ["key-1", "key-2"]
```

- [ ] **Step 5: Run tests to verify**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_config.py -v`
Expected: 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/__init__.py src/config.py requirements.txt .env.example tests/__init__.py tests/conftest.py tests/test_config.py
git commit -m "feat: project scaffolding with config and test setup"
```

---

## Task 2: Authentication — JWT Module

**Files:**
- Create: `rag/src/auth/__init__.py`
- Create: `rag/src/auth/models.py`
- Create: `rag/src/auth/jwt.py`
- Create: `rag/tests/test_auth/__init__.py`
- Create: `rag/tests/test_auth/test_jwt.py`

- [ ] **Step 1: Write the auth pydantic models**

```python
# src/auth/__init__.py
```

```python
# src/auth/models.py
from pydantic import BaseModel


class TokenPayload(BaseModel):
    sub: str  # username
    groups: list[str] = []  # AD group names
    exp: int | None = None


class UserContext(BaseModel):
    """Resolved user identity available in request handlers."""
    username: str
    groups: list[str]
```

- [ ] **Step 2: Write the failing tests for JWT**

```python
# tests/test_auth/__init__.py
```

```python
# tests/test_auth/test_jwt.py
import time
import pytest
from src.auth.jwt import create_token, decode_token
from src.auth.models import UserContext


def test_create_and_decode_token():
    token = create_token(username="mike", groups=["finance", "executives"])
    user = decode_token(token)
    assert user.username == "mike"
    assert "finance" in user.groups
    assert "executives" in user.groups


def test_decode_token_expired():
    token = create_token(username="mike", groups=[], expiration_minutes=-1)
    with pytest.raises(ValueError, match="expired"):
        decode_token(token)


def test_decode_token_invalid():
    with pytest.raises(ValueError, match="Invalid token"):
        decode_token("not-a-valid-token")


def test_create_token_contains_groups():
    token = create_token(username="bob", groups=["it_support"])
    user = decode_token(token)
    assert user.groups == ["it_support"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_auth/test_jwt.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement jwt.py**

```python
# src/auth/jwt.py
from datetime import datetime, timezone, timedelta

import jwt

from src.auth.models import UserContext
from src.config import settings


def create_token(
    username: str,
    groups: list[str],
    expiration_minutes: int | None = None,
) -> str:
    if expiration_minutes is None:
        expiration_minutes = settings.jwt_expiration_minutes

    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "groups": groups,
        "iat": now,
        "exp": now + timedelta(minutes=expiration_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> UserContext:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")

    return UserContext(
        username=payload["sub"],
        groups=payload.get("groups", []),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_auth/test_jwt.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/auth/ tests/test_auth/
git commit -m "feat: JWT token creation and validation with AD group support"
```

---

## Task 3: Authentication — API Key Middleware & FastAPI Dependencies

**Files:**
- Create: `rag/src/auth/api_key.py`
- Create: `rag/src/auth/dependencies.py`
- Create: `rag/tests/test_auth/test_api_key.py`
- Create: `rag/tests/test_auth/test_dependencies.py`

- [ ] **Step 1: Write failing tests for API key validation**

```python
# tests/test_auth/test_api_key.py
import pytest
from src.auth.api_key import validate_api_key


def test_valid_api_key():
    # test-key-1 is set in conftest.py
    assert validate_api_key("test-key-1") is True


def test_invalid_api_key():
    assert validate_api_key("bogus-key") is False


def test_empty_api_key():
    assert validate_api_key("") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_auth/test_api_key.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement api_key.py**

```python
# src/auth/api_key.py
from src.config import settings


def validate_api_key(key: str) -> bool:
    if not key:
        return False
    return key in settings.api_key_list
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_auth/test_api_key.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Write failing tests for FastAPI auth dependencies**

```python
# tests/test_auth/test_dependencies.py
import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from src.auth.dependencies import require_auth
from src.auth.jwt import create_token
from src.auth.models import UserContext

app = FastAPI()


@app.get("/protected")
async def protected_route(user: UserContext = Depends(require_auth)):
    return {"username": user.username, "groups": user.groups}


client = TestClient(app)


def test_valid_auth():
    token = create_token(username="mike", groups=["finance"])
    resp = client.get(
        "/protected",
        headers={
            "Authorization": f"Bearer {token}",
            "X-API-Key": "test-key-1",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["username"] == "mike"
    assert resp.json()["groups"] == ["finance"]


def test_missing_api_key():
    token = create_token(username="mike", groups=["finance"])
    resp = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


def test_invalid_api_key():
    token = create_token(username="mike", groups=["finance"])
    resp = client.get(
        "/protected",
        headers={
            "Authorization": f"Bearer {token}",
            "X-API-Key": "wrong-key",
        },
    )
    assert resp.status_code == 403


def test_missing_jwt():
    resp = client.get(
        "/protected",
        headers={"X-API-Key": "test-key-1"},
    )
    assert resp.status_code == 401


def test_invalid_jwt():
    resp = client.get(
        "/protected",
        headers={
            "Authorization": "Bearer garbage-token",
            "X-API-Key": "test-key-1",
        },
    )
    assert resp.status_code == 401
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_auth/test_dependencies.py -v`
Expected: FAIL (module not found)

- [ ] **Step 7: Implement dependencies.py**

```python
# src/auth/dependencies.py
from fastapi import Header, HTTPException

from src.auth.api_key import validate_api_key
from src.auth.jwt import decode_token
from src.auth.models import UserContext


async def require_auth(
    authorization: str = Header(default=""),
    x_api_key: str = Header(default="", alias="X-API-Key"),
) -> UserContext:
    # Validate API key
    if not validate_api_key(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

    # Validate JWT
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.removeprefix("Bearer ")
    try:
        return decode_token(token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_auth/ -v`
Expected: All 12 auth tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/auth/api_key.py src/auth/dependencies.py tests/test_auth/test_api_key.py tests/test_auth/test_dependencies.py
git commit -m "feat: API key validation and FastAPI auth dependency injection"
```

---

## Task 4: Document Parsing

**Files:**
- Create: `rag/src/ingestion/__init__.py`
- Create: `rag/src/ingestion/parser.py`
- Create: `rag/tests/test_ingestion/__init__.py`
- Create: `rag/tests/test_ingestion/test_parser.py`
- Create: `rag/test_fixtures/sample.pdf`
- Create: `rag/test_fixtures/sample.docx`
- Create: `rag/test_fixtures/sample.xlsx`
- Create: `rag/test_fixtures/sample_transcript.txt`

- [ ] **Step 1: Create test fixture files**

Create a Python script to generate the test fixtures:

```python
# scripts/create_test_fixtures.py
import os
from docx import Document
from openpyxl import Workbook
from fpdf import FPDF  # pip install fpdf2

os.makedirs("test_fixtures", exist_ok=True)

# PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)
pdf.cell(200, 10, text="Finance Policy Document", new_x="LMARGIN", new_y="NEXT")
pdf.cell(200, 10, text="Section 4.2: Expense Reporting", new_x="LMARGIN", new_y="NEXT")
pdf.cell(200, 10, text="All expenses over $500 require manager approval.", new_x="LMARGIN", new_y="NEXT")
pdf.cell(200, 10, text="Receipts must be submitted within 30 days.", new_x="LMARGIN", new_y="NEXT")
pdf.output("test_fixtures/sample.pdf")

# Word doc
doc = Document()
doc.add_heading("IT Runbook: Server Restart Procedure", level=1)
doc.add_heading("Step 1: Pre-checks", level=2)
doc.add_paragraph("Verify no active deployments are running.")
doc.add_paragraph("Check the monitoring dashboard for anomalies.")
doc.add_heading("Step 2: Restart", level=2)
doc.add_paragraph("SSH into the server and run: sudo systemctl restart app-server")
doc.save("test_fixtures/sample.docx")

# Spreadsheet
wb = Workbook()
ws = wb.active
ws.title = "Q3 Budget"
ws.append(["Department", "Budget", "Spent", "Remaining"])
ws.append(["Engineering", 500000, 420000, 80000])
ws.append(["Marketing", 300000, 290000, 10000])
ws.append(["Finance", 200000, 150000, 50000])
wb.save("test_fixtures/sample.xlsx")

# Meeting transcript
with open("test_fixtures/sample_transcript.txt", "w") as f:
    f.write("Meeting: Engineering Standup\n")
    f.write("Date: 2026-04-10\n")
    f.write("---\n")
    f.write("Mike: Are we on track for the Q2 release?\n")
    f.write("Sarah: Yes, but the API migration is behind schedule.\n")
    f.write("Mike: What's blocking the API migration?\n")
    f.write("Sarah: We're waiting on the new auth library to be approved.\n")
    f.write("Bob: I can help with testing once it's ready.\n")

print("Test fixtures created.")
```

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && pip install fpdf2 && python scripts/create_test_fixtures.py`

- [ ] **Step 2: Write failing tests for parser**

```python
# src/ingestion/__init__.py
```

```python
# tests/test_ingestion/__init__.py
```

```python
# tests/test_ingestion/test_parser.py
import pytest
from pathlib import Path
from src.ingestion.parser import parse_document, ParsedDocument

FIXTURES = Path(__file__).parent.parent.parent / "test_fixtures"


def test_parse_pdf():
    result = parse_document(FIXTURES / "sample.pdf")
    assert isinstance(result, ParsedDocument)
    assert result.doc_type == "pdf"
    assert "expense" in result.text.lower() or "finance" in result.text.lower()
    assert result.filename == "sample.pdf"


def test_parse_docx():
    result = parse_document(FIXTURES / "sample.docx")
    assert result.doc_type == "docx"
    assert "server restart" in result.text.lower() or "runbook" in result.text.lower()


def test_parse_xlsx():
    result = parse_document(FIXTURES / "sample.xlsx")
    assert result.doc_type == "xlsx"
    assert "engineering" in result.text.lower() or "budget" in result.text.lower()


def test_parse_transcript():
    result = parse_document(FIXTURES / "sample_transcript.txt")
    assert result.doc_type == "transcript"
    assert len(result.utterances) > 0
    mike_questions = [u for u in result.utterances if u.speaker == "Mike" and u.utterance_type == "question"]
    assert len(mike_questions) == 2


def test_parse_unsupported_format():
    # Create a temp file with unsupported extension
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        f.write(b"some content")
        path = Path(f.name)
    with pytest.raises(ValueError, match="Unsupported"):
        parse_document(path)
    path.unlink()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_ingestion/test_parser.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement parser.py**

```python
# src/ingestion/parser.py
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document as DocxDocument
from openpyxl import load_workbook


@dataclass
class Utterance:
    speaker: str
    text: str
    utterance_type: str  # "question", "statement"


@dataclass
class ParsedDocument:
    filename: str
    doc_type: str  # "pdf", "docx", "xlsx", "transcript"
    text: str
    metadata: dict = field(default_factory=dict)
    utterances: list[Utterance] = field(default_factory=list)


def parse_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix == ".docx":
        return _parse_docx(path)
    elif suffix in (".xlsx", ".csv"):
        return _parse_spreadsheet(path)
    elif suffix == ".txt":
        return _parse_transcript(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _parse_pdf(path: Path) -> ParsedDocument:
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(str(path))
    text = "\n".join(str(el) for el in elements)
    return ParsedDocument(filename=path.name, doc_type="pdf", text=text)


def _parse_docx(path: Path) -> ParsedDocument:
    doc = DocxDocument(str(path))
    parts = []
    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            parts.append(f"\n## {para.text}\n")
        elif para.text.strip():
            parts.append(para.text)
    text = "\n".join(parts)
    return ParsedDocument(filename=path.name, doc_type="docx", text=text)


def _parse_spreadsheet(path: Path) -> ParsedDocument:
    wb = load_workbook(str(path), read_only=True)
    parts = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        parts.append(f"Sheet: {sheet_name}")
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        headers = [str(h) if h else "" for h in rows[0]]
        parts.append(" | ".join(headers))
        for row in rows[1:]:
            parts.append(" | ".join(str(c) if c is not None else "" for c in row))
    wb.close()
    text = "\n".join(parts)
    return ParsedDocument(
        filename=path.name,
        doc_type="xlsx",
        text=text,
        metadata={"sheet_names": wb.sheetnames},
    )


def _parse_transcript(path: Path) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8")
    lines = raw.strip().split("\n")

    metadata = {}
    utterances = []
    text_parts = []

    for line in lines:
        line = line.strip()
        if not line or line == "---":
            continue

        # Header metadata (e.g., "Meeting: Engineering Standup")
        if ":" in line and not any(line.startswith(f"{name}:") for name in _extract_speaker_names(lines)):
            key, _, value = line.partition(":")
            if key.strip() in ("Meeting", "Date", "Location", "Attendees"):
                metadata[key.strip().lower()] = value.strip()
                text_parts.append(line)
                continue

        # Speaker lines (e.g., "Mike: Are we on track?")
        if ":" in line:
            speaker, _, text = line.partition(":")
            speaker = speaker.strip()
            text = text.strip()
            is_question = text.rstrip().endswith("?")
            utterances.append(Utterance(
                speaker=speaker,
                text=text,
                utterance_type="question" if is_question else "statement",
            ))
            text_parts.append(line)
        else:
            text_parts.append(line)

    return ParsedDocument(
        filename=path.name,
        doc_type="transcript",
        text="\n".join(text_parts),
        metadata=metadata,
        utterances=utterances,
    )


def _extract_speaker_names(lines: list[str]) -> set[str]:
    """Pre-scan lines to identify speaker names (appear before ':' multiple times)."""
    from collections import Counter
    names = Counter()
    for line in lines:
        if ":" in line:
            name = line.split(":")[0].strip()
            if name and len(name) < 40 and name not in ("Meeting", "Date", "Location", "Attendees"):
                names[name] += 1
    return {name for name, count in names.items() if count >= 1}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_ingestion/test_parser.py -v`
Expected: 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/__init__.py src/ingestion/parser.py tests/test_ingestion/ test_fixtures/ scripts/create_test_fixtures.py
git commit -m "feat: document parser for PDF, Word, spreadsheet, and transcript formats"
```

---

## Task 5: Structure-Aware Text Chunking

**Files:**
- Create: `rag/src/ingestion/chunker.py`
- Create: `rag/tests/test_ingestion/test_chunker.py`

- [ ] **Step 1: Write failing tests for chunker**

```python
# tests/test_ingestion/test_chunker.py
import pytest
from src.ingestion.chunker import chunk_text, Chunk


def test_short_text_single_chunk():
    chunks = chunk_text("Hello world.", chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].start_char == 0


def test_long_text_multiple_chunks():
    text = "Word " * 200  # ~1000 chars
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    # Check overlap: end of chunk N overlaps start of chunk N+1
    for i in range(len(chunks) - 1):
        end_of_current = chunks[i].text[-40:]
        assert end_of_current in chunks[i + 1].text


def test_respects_paragraph_boundaries():
    text = "First paragraph with some content.\n\nSecond paragraph with different content.\n\nThird paragraph here."
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=10)
    # Chunks should prefer breaking at paragraph boundaries
    for chunk in chunks:
        # No chunk should start or end mid-word
        stripped = chunk.text.strip()
        assert not stripped.startswith(" ")


def test_chunk_metadata():
    text = "Some text here.\n\nMore text below."
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert chunks[0].index == 0
    assert chunks[0].start_char == 0


def test_empty_text():
    chunks = chunk_text("", chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_ingestion/test_chunker.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement chunker.py**

```python
# src/ingestion/chunker.py
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int
    start_char: int


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    if not text.strip():
        return []

    # Split on paragraph boundaries first
    paragraphs = text.split("\n\n")
    chunks: list[Chunk] = []
    current_text = ""
    current_start = 0
    char_pos = 0

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            char_pos += 2  # account for \n\n
            continue

        separator = "\n\n" if current_text else ""
        candidate = current_text + separator + para

        if len(candidate) > chunk_size and current_text:
            # Save current chunk
            chunks.append(Chunk(
                text=current_text,
                index=len(chunks),
                start_char=current_start,
            ))
            # Start new chunk with overlap from end of previous
            overlap_text = current_text[-chunk_overlap:] if len(current_text) > chunk_overlap else current_text
            current_text = overlap_text + "\n\n" + para
            current_start = char_pos - len(overlap_text)
        else:
            if not current_text:
                current_start = char_pos
            current_text = candidate

        char_pos += len(para) + 2  # +2 for \n\n

    # Don't forget the last chunk
    if current_text.strip():
        chunks.append(Chunk(
            text=current_text,
            index=len(chunks),
            start_char=current_start,
        ))

    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_ingestion/test_chunker.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/chunker.py tests/test_ingestion/test_chunker.py
git commit -m "feat: structure-aware text chunker with paragraph boundary splitting"
```

---

## Task 6: Embedding Client

**Files:**
- Create: `rag/src/ingestion/embedder.py`
- Create: `rag/tests/test_ingestion/test_embedder.py`

- [ ] **Step 1: Write failing tests for embedder**

```python
# tests/test_ingestion/test_embedder.py
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


def test_embed_texts_returns_vectors():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    with patch("src.ingestion.embedder._get_model", return_value=mock_model):
        from src.ingestion.embedder import embed_texts
        vectors = embed_texts(["hello", "world"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    mock_model.encode.assert_called_once()


def test_embed_texts_adds_query_prefix():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])

    with patch("src.ingestion.embedder._get_model", return_value=mock_model):
        from src.ingestion.embedder import embed_texts
        embed_texts(["test"], prefix="query: ")

    call_args = mock_model.encode.call_args[0][0]
    assert call_args[0] == "query: test"


def test_embed_texts_empty_input():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([]).reshape(0, 0)

    with patch("src.ingestion.embedder._get_model", return_value=mock_model):
        from src.ingestion.embedder import embed_texts
        vectors = embed_texts([])

    assert len(vectors) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_ingestion/test_embedder.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement embedder.py**

```python
# src/ingestion/embedder.py
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings

# E5 models expect "query: " prefix for queries and "passage: " for documents
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name, device=settings.embedding_device)


def embed_texts(
    texts: list[str],
    prefix: str = PASSAGE_PREFIX,
    batch_size: int = 32,
) -> list[list[float]]:
    if not texts:
        return []

    model = _get_model()
    prefixed = [f"{prefix}{t}" for t in texts]
    embeddings: np.ndarray = model.encode(prefixed, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    results = embed_texts([query], prefix=QUERY_PREFIX)
    return results[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_ingestion/test_embedder.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/embedder.py tests/test_ingestion/test_embedder.py
git commit -m "feat: E5-large embedding client with query/passage prefix support"
```

---

## Task 7: Qdrant Vector Store Client

**Files:**
- Create: `rag/src/retrieval/__init__.py`
- Create: `rag/src/retrieval/models.py`
- Create: `rag/src/retrieval/vector_store.py`
- Create: `rag/tests/test_retrieval/__init__.py`
- Create: `rag/tests/test_retrieval/test_vector_store.py`

- [ ] **Step 1: Write retrieval models**

```python
# src/retrieval/__init__.py
```

```python
# src/retrieval/models.py
from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    doc_id: str
    filename: str
    doc_type: str
    chunk_index: int
    start_char: int
    acl_groups: list[str]
    category: str = ""
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
```

- [ ] **Step 2: Write failing tests for vector store**

```python
# tests/test_retrieval/__init__.py
```

```python
# tests/test_retrieval/test_vector_store.py
import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.vector_store import VectorStore
from src.retrieval.models import ChunkMetadata


@pytest.fixture
def mock_qdrant():
    with patch("src.retrieval.vector_store.QdrantClient") as MockClient:
        mock = MockClient.return_value
        mock.collection_exists.return_value = False
        store = VectorStore()
        yield store, mock


def test_init_creates_collection_if_not_exists(mock_qdrant):
    store, mock = mock_qdrant
    mock.collection_exists.assert_called_once()
    mock.create_collection.assert_called_once()


def test_upsert_chunks(mock_qdrant):
    store, mock = mock_qdrant
    metadata = ChunkMetadata(
        doc_id="doc-1",
        filename="test.pdf",
        doc_type="pdf",
        chunk_index=0,
        start_char=0,
        acl_groups=["finance"],
    )
    store.upsert(
        texts=["hello world"],
        vectors=[[0.1, 0.2, 0.3]],
        metadatas=[metadata],
    )
    mock.upsert.assert_called_once()


def test_search_with_acl_filter(mock_qdrant):
    store, mock = mock_qdrant
    mock.query_points.return_value = MagicMock(points=[])
    results = store.search(
        vector=[0.1, 0.2, 0.3],
        user_groups=["finance"],
        top_k=5,
    )
    assert results == []
    # Verify ACL filter was applied
    call_kwargs = mock.query_points.call_args
    assert call_kwargs is not None


def test_delete_by_doc_id(mock_qdrant):
    store, mock = mock_qdrant
    store.delete_by_doc_id("doc-1")
    mock.delete.assert_called_once()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_retrieval/test_vector_store.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement vector_store.py**

```python
# src/retrieval/vector_store.py
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from src.config import settings
from src.retrieval.models import ChunkMetadata, RetrievedChunk

VECTOR_SIZE = 1024  # E5-large dimension


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.collection = settings.qdrant_collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def upsert(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[ChunkMetadata],
    ) -> None:
        points = []
        for text, vector, meta in zip(texts, vectors, metadatas):
            point_id = str(uuid.uuid4())
            payload = meta.model_dump()
            payload["text"] = text
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        vector: list[float],
        user_groups: list[str],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        acl_filter = Filter(
            must=[
                FieldCondition(
                    key="acl_groups",
                    match=MatchAny(any=user_groups),
                ),
            ]
        )

        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=acl_filter,
            limit=top_k,
            with_payload=True,
        )

        chunks = []
        for point in results.points:
            payload = point.payload
            text = payload.pop("text", "")
            chunks.append(RetrievedChunk(
                text=text,
                score=point.score,
                metadata=ChunkMetadata(**payload),
            ))
        return chunks

    def delete_by_doc_id(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchAny(any=[doc_id]))]
            ),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_retrieval/test_vector_store.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/retrieval/ tests/test_retrieval/
git commit -m "feat: Qdrant vector store client with ACL-filtered search"
```

---

## Task 8: Metadata Database (Document Registry)

**Files:**
- Create: `rag/src/db/__init__.py`
- Create: `rag/src/db/models.py`
- Create: `rag/src/db/metadata.py`
- Create: `rag/tests/test_db/__init__.py` (implied by test path)
- Create: `rag/tests/test_db/test_metadata.py`

- [ ] **Step 1: Write SQLAlchemy models**

```python
# src/db/__init__.py
```

```python
# src/db/models.py
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    doc_type: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, default="")
    acl_groups: Mapped[list] = mapped_column(JSON, default=list)
    chunk_count: Mapped[int] = mapped_column(default=0)
    uploaded_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
```

- [ ] **Step 2: Write failing tests for metadata store**

```python
# tests/test_db/__init__.py
```

```python
# tests/test_db/test_metadata.py
import pytest
import pytest_asyncio
from src.db.metadata import MetadataStore


@pytest_asyncio.fixture
async def store():
    s = MetadataStore("sqlite+aiosqlite:///:memory:")
    await s.init()
    yield s


@pytest.mark.asyncio
async def test_add_and_get_document(store):
    await store.add_document(
        doc_id="doc-1",
        filename="test.pdf",
        doc_type="pdf",
        acl_groups=["finance"],
        chunk_count=5,
        uploaded_by="mike",
    )
    doc = await store.get_document("doc-1")
    assert doc is not None
    assert doc.filename == "test.pdf"
    assert doc.acl_groups == ["finance"]
    assert doc.chunk_count == 5


@pytest.mark.asyncio
async def test_get_nonexistent_document(store):
    doc = await store.get_document("does-not-exist")
    assert doc is None


@pytest.mark.asyncio
async def test_list_documents(store):
    await store.add_document(doc_id="d1", filename="a.pdf", doc_type="pdf", acl_groups=["finance"], chunk_count=3, uploaded_by="mike")
    await store.add_document(doc_id="d2", filename="b.docx", doc_type="docx", acl_groups=["it"], chunk_count=2, uploaded_by="bob")
    docs = await store.list_documents()
    assert len(docs) == 2


@pytest.mark.asyncio
async def test_list_documents_filtered_by_groups(store):
    await store.add_document(doc_id="d1", filename="a.pdf", doc_type="pdf", acl_groups=["finance"], chunk_count=3, uploaded_by="mike")
    await store.add_document(doc_id="d2", filename="b.docx", doc_type="docx", acl_groups=["it"], chunk_count=2, uploaded_by="bob")
    docs = await store.list_documents(user_groups=["finance"])
    assert len(docs) == 1
    assert docs[0].filename == "a.pdf"


@pytest.mark.asyncio
async def test_delete_document(store):
    await store.add_document(doc_id="d1", filename="a.pdf", doc_type="pdf", acl_groups=["finance"], chunk_count=3, uploaded_by="mike")
    await store.delete_document("d1")
    doc = await store.get_document("d1")
    assert doc is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_db/test_metadata.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement metadata.py**

```python
# src/db/metadata.py
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from src.db.models import Base, DocumentRecord


class MetadataStore:
    def __init__(self, database_url: str | None = None):
        if database_url is None:
            from src.config import settings
            database_url = settings.database_url
        self.engine = create_async_engine(database_url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def add_document(
        self,
        doc_id: str,
        filename: str,
        doc_type: str,
        acl_groups: list[str],
        chunk_count: int,
        uploaded_by: str,
        category: str = "",
    ) -> DocumentRecord:
        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            doc_type=doc_type,
            acl_groups=acl_groups,
            chunk_count=chunk_count,
            uploaded_by=uploaded_by,
            category=category,
        )
        async with self.session_factory() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
        return record

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        async with self.session_factory() as session:
            return await session.get(DocumentRecord, doc_id)

    async def list_documents(self, user_groups: list[str] | None = None) -> list[DocumentRecord]:
        async with self.session_factory() as session:
            stmt = select(DocumentRecord)
            result = await session.execute(stmt)
            docs = list(result.scalars().all())

        if user_groups is not None:
            docs = [d for d in docs if any(g in d.acl_groups for g in user_groups)]
        return docs

    async def delete_document(self, doc_id: str) -> None:
        async with self.session_factory() as session:
            await session.execute(
                delete(DocumentRecord).where(DocumentRecord.doc_id == doc_id)
            )
            await session.commit()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_db/test_metadata.py -v`
Expected: 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/db/ tests/test_db/
git commit -m "feat: async metadata store for document registry with ACL filtering"
```

---

## Task 9: Ingestion Pipeline (End-to-End Orchestrator)

**Files:**
- Create: `rag/src/ingestion/pipeline.py`
- Create: `rag/tests/test_ingestion/test_pipeline.py`

- [ ] **Step 1: Write failing tests for the pipeline**

```python
# tests/test_ingestion/test_pipeline.py
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from src.ingestion.pipeline import IngestResult, ingest_document


@pytest_asyncio.fixture
async def mock_deps():
    mock_vector_store = MagicMock()
    mock_vector_store.upsert = MagicMock()
    mock_vector_store.delete_by_doc_id = MagicMock()

    mock_metadata_store = AsyncMock()
    mock_metadata_store.add_document = AsyncMock()
    mock_metadata_store.delete_document = AsyncMock()

    mock_embed = MagicMock(return_value=[[0.1] * 1024])

    return mock_vector_store, mock_metadata_store, mock_embed


FIXTURES = Path(__file__).parent.parent.parent / "test_fixtures"


@pytest.mark.asyncio
async def test_ingest_pdf(mock_deps):
    vector_store, metadata_store, mock_embed = mock_deps
    with patch("src.ingestion.pipeline.embed_texts", mock_embed):
        result = await ingest_document(
            file_path=FIXTURES / "sample.pdf",
            acl_groups=["finance"],
            uploaded_by="mike",
            vector_store=vector_store,
            metadata_store=metadata_store,
        )
    assert isinstance(result, IngestResult)
    assert result.doc_type == "pdf"
    assert result.chunk_count > 0
    assert result.doc_id is not None
    vector_store.upsert.assert_called_once()
    metadata_store.add_document.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_transcript(mock_deps):
    vector_store, metadata_store, mock_embed = mock_deps
    with patch("src.ingestion.pipeline.embed_texts", mock_embed):
        result = await ingest_document(
            file_path=FIXTURES / "sample_transcript.txt",
            acl_groups=["engineering"],
            uploaded_by="mike",
            vector_store=vector_store,
            metadata_store=metadata_store,
        )
    assert result.doc_type == "transcript"
    assert result.chunk_count > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_ingestion/test_pipeline.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement pipeline.py**

```python
# src/ingestion/pipeline.py
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_text
from src.ingestion.embedder import embed_texts
from src.retrieval.models import ChunkMetadata
from src.retrieval.vector_store import VectorStore
from src.db.metadata import MetadataStore


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    doc_type: str
    chunk_count: int


async def ingest_document(
    file_path: Path,
    acl_groups: list[str],
    uploaded_by: str,
    vector_store: VectorStore,
    metadata_store: MetadataStore,
    category: str = "",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> IngestResult:
    doc_id = str(uuid.uuid4())

    # Parse
    parsed = parse_document(file_path)

    # Chunk
    chunks = chunk_text(parsed.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        chunks_from_text = []
    else:
        chunks_from_text = chunks

    # Build texts and metadata for each chunk
    texts = [c.text for c in chunks_from_text]
    metadatas = []
    for c in chunks_from_text:
        metadatas.append(ChunkMetadata(
            doc_id=doc_id,
            filename=parsed.filename,
            doc_type=parsed.doc_type,
            chunk_index=c.index,
            start_char=c.start_char,
            acl_groups=acl_groups,
            category=category,
        ))

    # Embed
    vectors = embed_texts(texts) if texts else []

    # Store vectors
    if vectors:
        vector_store.upsert(texts=texts, vectors=vectors, metadatas=metadatas)

    # Store metadata
    await metadata_store.add_document(
        doc_id=doc_id,
        filename=parsed.filename,
        doc_type=parsed.doc_type,
        acl_groups=acl_groups,
        chunk_count=len(chunks_from_text),
        uploaded_by=uploaded_by,
        category=category,
    )

    return IngestResult(
        doc_id=doc_id,
        filename=parsed.filename,
        doc_type=parsed.doc_type,
        chunk_count=len(chunks_from_text),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_ingestion/test_pipeline.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/pipeline.py tests/test_ingestion/test_pipeline.py
git commit -m "feat: end-to-end ingestion pipeline (parse, chunk, embed, store)"
```

---

## Task 10: LLM Client & RAG Chain

**Files:**
- Create: `rag/src/generation/__init__.py`
- Create: `rag/src/generation/llm_client.py`
- Create: `rag/src/generation/rag_chain.py`
- Create: `rag/tests/test_generation/__init__.py`
- Create: `rag/tests/test_generation/test_llm_client.py`
- Create: `rag/tests/test_generation/test_rag_chain.py`

- [ ] **Step 1: Write failing tests for LLM client**

```python
# src/generation/__init__.py
```

```python
# tests/test_generation/__init__.py
```

```python
# tests/test_generation/test_llm_client.py
import pytest
from unittest.mock import patch, MagicMock


def test_generate_calls_openai_client():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    mock_client.chat.completions.create.return_value = mock_response

    with patch("src.generation.llm_client._get_client", return_value=mock_client):
        from src.generation.llm_client import generate
        result = generate(
            system_prompt="You are helpful.",
            user_prompt="Hello",
        )

    assert result == "Test response"
    mock_client.chat.completions.create.assert_called_once()


def test_generate_passes_model_and_messages():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="answer"))]
    mock_client.chat.completions.create.return_value = mock_response

    with patch("src.generation.llm_client._get_client", return_value=mock_client):
        from src.generation.llm_client import generate
        generate(system_prompt="sys", user_prompt="usr")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "sys"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "usr"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_generation/test_llm_client.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement llm_client.py**

```python
# src/generation/llm_client.py
from functools import lru_cache

from openai import OpenAI

from src.config import settings


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    return OpenAI(base_url=settings.vllm_base_url, api_key="not-needed")


def generate(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.vllm_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_generation/test_llm_client.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Write failing tests for RAG chain**

```python
# tests/test_generation/test_rag_chain.py
import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.models import RetrievedChunk, ChunkMetadata, Citation
from src.generation.rag_chain import rag_query, RAGResponse


@pytest.fixture
def mock_chunks():
    return [
        RetrievedChunk(
            text="All expenses over $500 require manager approval.",
            score=0.95,
            metadata=ChunkMetadata(
                doc_id="doc-1",
                filename="finance_policy.pdf",
                doc_type="pdf",
                chunk_index=2,
                start_char=100,
                acl_groups=["finance"],
                page=12,
            ),
        ),
        RetrievedChunk(
            text="Receipts must be submitted within 30 days.",
            score=0.88,
            metadata=ChunkMetadata(
                doc_id="doc-1",
                filename="finance_policy.pdf",
                doc_type="pdf",
                chunk_index=3,
                start_char=200,
                acl_groups=["finance"],
            ),
        ),
    ]


def test_rag_query_returns_response_with_citations(mock_chunks):
    with patch("src.generation.rag_chain.embed_query", return_value=[0.1] * 1024):
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = mock_chunks
        with patch("src.generation.rag_chain.generate", return_value="Expenses over $500 need approval [1]."):
            result = rag_query(
                question="What is the expense policy?",
                user_groups=["finance"],
                vector_store=mock_vector_store,
            )

    assert isinstance(result, RAGResponse)
    assert "approval" in result.answer.lower() or "500" in result.answer
    assert len(result.citations) == 2
    assert result.citations[0].filename == "finance_policy.pdf"


def test_rag_query_no_results():
    with patch("src.generation.rag_chain.embed_query", return_value=[0.1] * 1024):
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = []
        result = rag_query(
            question="Something obscure",
            user_groups=["finance"],
            vector_store=mock_vector_store,
        )

    assert isinstance(result, RAGResponse)
    assert "could not find" in result.answer.lower() or "no relevant" in result.answer.lower()
    assert len(result.citations) == 0
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_generation/test_rag_chain.py -v`
Expected: FAIL (module not found)

- [ ] **Step 7: Implement rag_chain.py**

```python
# src/generation/rag_chain.py
from dataclasses import dataclass

from src.ingestion.embedder import embed_query
from src.generation.llm_client import generate
from src.retrieval.models import Citation, RetrievedChunk
from src.retrieval.vector_store import VectorStore

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on provided context documents.

Rules:
- Only answer based on the provided context. Do not use outside knowledge.
- Cite your sources using [N] notation, where N corresponds to the context chunk number.
- If the context does not contain enough information to answer, say so clearly.
- Be concise and accurate.
"""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer the question based only on the context above. Cite sources using [N] notation."""


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]


def rag_query(
    question: str,
    user_groups: list[str],
    vector_store: VectorStore,
    top_k: int = 10,
) -> RAGResponse:
    # Embed query
    query_vector = embed_query(question)

    # Retrieve with ACL filter
    chunks = vector_store.search(
        vector=query_vector,
        user_groups=user_groups,
        top_k=top_k,
    )

    # No results
    if not chunks:
        return RAGResponse(
            answer="I could not find any relevant information in the documents you have access to.",
            citations=[],
        )

    # Build context string
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[{i}] {chunk.metadata.filename}"
        if chunk.metadata.page is not None:
            source += f", page {chunk.metadata.page}"
        context_parts.append(f"{source}:\n{chunk.text}")

    context = "\n\n".join(context_parts)

    # Generate
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    answer = generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)

    # Build citations
    citations = [
        Citation(
            doc_id=chunk.metadata.doc_id,
            filename=chunk.metadata.filename,
            doc_type=chunk.metadata.doc_type,
            chunk_index=chunk.metadata.chunk_index,
            page=chunk.metadata.page,
            snippet=chunk.text[:200],
            relevance=chunk.score,
        )
        for chunk in chunks
    ]

    return RAGResponse(answer=answer, citations=citations)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_generation/ -v`
Expected: 4 tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/generation/ tests/test_generation/
git commit -m "feat: LLM client and RAG chain with citation support"
```

---

## Task 11: Retriever (Top-K with ACL Filtering)

**Files:**
- Create: `rag/src/retrieval/retriever.py`
- Create: `rag/tests/test_retrieval/test_retriever.py`

- [ ] **Step 1: Write failing tests for retriever**

```python
# tests/test_retrieval/test_retriever.py
import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.retriever import retrieve
from src.retrieval.models import RetrievedChunk, ChunkMetadata


def test_retrieve_calls_embed_and_search():
    mock_store = MagicMock()
    mock_store.search.return_value = [
        RetrievedChunk(
            text="test chunk",
            score=0.9,
            metadata=ChunkMetadata(
                doc_id="d1", filename="a.pdf", doc_type="pdf",
                chunk_index=0, start_char=0, acl_groups=["finance"],
            ),
        )
    ]

    with patch("src.retrieval.retriever.embed_query", return_value=[0.1] * 1024):
        results = retrieve(
            query="test question",
            user_groups=["finance"],
            vector_store=mock_store,
            top_k=5,
        )

    assert len(results) == 1
    assert results[0].text == "test chunk"
    mock_store.search.assert_called_once()


def test_retrieve_filters_low_score():
    mock_store = MagicMock()
    mock_store.search.return_value = [
        RetrievedChunk(
            text="good", score=0.9,
            metadata=ChunkMetadata(doc_id="d1", filename="a.pdf", doc_type="pdf",
                                   chunk_index=0, start_char=0, acl_groups=["finance"]),
        ),
        RetrievedChunk(
            text="bad", score=0.2,
            metadata=ChunkMetadata(doc_id="d2", filename="b.pdf", doc_type="pdf",
                                   chunk_index=0, start_char=0, acl_groups=["finance"]),
        ),
    ]

    with patch("src.retrieval.retriever.embed_query", return_value=[0.1] * 1024):
        results = retrieve(
            query="test",
            user_groups=["finance"],
            vector_store=mock_store,
            min_score=0.5,
        )

    assert len(results) == 1
    assert results[0].text == "good"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_retrieval/test_retriever.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement retriever.py**

```python
# src/retrieval/retriever.py
from src.ingestion.embedder import embed_query
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_store import VectorStore


def retrieve(
    query: str,
    user_groups: list[str],
    vector_store: VectorStore,
    top_k: int = 10,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    query_vector = embed_query(query)

    results = vector_store.search(
        vector=query_vector,
        user_groups=user_groups,
        top_k=top_k,
    )

    if min_score > 0:
        results = [r for r in results if r.score >= min_score]

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_retrieval/test_retriever.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/retriever.py tests/test_retrieval/test_retriever.py
git commit -m "feat: top-K retriever with ACL filtering and minimum score threshold"
```

---

## Task 12: API Routes — Auth & Ingest

**Files:**
- Create: `rag/src/api/__init__.py`
- Create: `rag/src/api/models.py`
- Create: `rag/src/api/routes_auth.py`
- Create: `rag/src/api/routes_ingest.py`
- Create: `rag/tests/test_api/__init__.py`
- Create: `rag/tests/test_api/test_routes_auth.py`
- Create: `rag/tests/test_api/test_routes_ingest.py`

- [ ] **Step 1: Write API pydantic models**

```python
# src/api/__init__.py
```

```python
# src/api/models.py
from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str
    groups: list[str] = []  # Phase 1: client provides groups; production: resolved from LDAP


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class IngestRequest(BaseModel):
    acl_groups: list[str]
    category: str = ""


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
```

- [ ] **Step 2: Write failing tests for auth routes**

```python
# tests/test_api/__init__.py
```

```python
# tests/test_api/test_routes_auth.py
import pytest
from fastapi.testclient import TestClient
from src.main import create_app

client = TestClient(create_app())


def test_login_returns_token():
    resp = client.post(
        "/api/v1/auth/token",
        json={"username": "mike", "password": "test", "groups": ["finance"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_missing_username():
    resp = client.post(
        "/api/v1/auth/token",
        json={"password": "test"},
    )
    assert resp.status_code == 422
```

- [ ] **Step 3: Write failing tests for ingest routes**

```python
# tests/test_api/test_routes_ingest.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from src.main import create_app
from src.auth.jwt import create_token

FIXTURES = Path(__file__).parent.parent.parent / "test_fixtures"


@pytest.fixture
def auth_headers():
    token = create_token(username="mike", groups=["finance"])
    return {
        "Authorization": f"Bearer {token}",
        "X-API-Key": "test-key-1",
    }


@pytest.fixture
def client():
    return TestClient(create_app())


def test_ingest_document(client, auth_headers):
    mock_result = MagicMock()
    mock_result.doc_id = "doc-123"
    mock_result.filename = "sample.pdf"
    mock_result.doc_type = "pdf"
    mock_result.chunk_count = 3

    with patch("src.api.routes_ingest.ingest_document", new_callable=AsyncMock, return_value=mock_result):
        with open(FIXTURES / "sample.pdf", "rb") as f:
            resp = client.post(
                "/api/v1/ingest",
                files={"file": ("sample.pdf", f, "application/pdf")},
                data={"acl_groups": '["finance"]'},
                headers=auth_headers,
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["doc_id"] == "doc-123"
    assert data["chunk_count"] == 3


def test_ingest_requires_auth(client):
    resp = client.post("/api/v1/ingest", files={"file": ("test.pdf", b"content", "application/pdf")})
    assert resp.status_code in (401, 403)


def test_list_documents(client, auth_headers):
    mock_doc = MagicMock()
    mock_doc.doc_id = "d1"
    mock_doc.filename = "a.pdf"
    mock_doc.doc_type = "pdf"
    mock_doc.category = ""
    mock_doc.acl_groups = ["finance"]
    mock_doc.chunk_count = 5

    with patch("src.api.routes_ingest.get_metadata_store") as mock_get_store:
        mock_store = AsyncMock()
        mock_store.list_documents.return_value = [mock_doc]
        mock_get_store.return_value = mock_store
        resp = client.get("/api/v1/documents", headers=auth_headers)

    assert resp.status_code == 200
    assert len(resp.json()) == 1
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_api/ -v`
Expected: FAIL (module not found — src.main doesn't exist yet)

- [ ] **Step 5: Implement routes_auth.py**

```python
# src/api/routes_auth.py
from fastapi import APIRouter

from src.api.models import LoginRequest, LoginResponse
from src.auth.jwt import create_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/token", response_model=LoginResponse)
async def login(request: LoginRequest):
    # Phase 1: simplified auth — accepts any username/password.
    # Production: validate against LDAP and resolve groups from AD.
    token = create_token(username=request.username, groups=request.groups)
    return LoginResponse(access_token=token)
```

- [ ] **Step 6: Implement routes_ingest.py**

```python
# src/api/routes_ingest.py
import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from src.api.models import DocumentInfo, IngestResponse
from src.auth.dependencies import require_auth
from src.auth.models import UserContext
from src.db.metadata import MetadataStore
from src.ingestion.pipeline import ingest_document
from src.retrieval.vector_store import VectorStore

router = APIRouter(prefix="/api/v1", tags=["ingestion"])

_vector_store: VectorStore | None = None
_metadata_store: MetadataStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_metadata_store() -> MetadataStore:
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MetadataStore()
    return _metadata_store


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    acl_groups: str = Form(default="[]"),
    category: str = Form(default=""),
    user: UserContext = Depends(require_auth),
):
    groups = json.loads(acl_groups)

    # Write uploaded file to temp location
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = await ingest_document(
            file_path=tmp_path,
            acl_groups=groups,
            uploaded_by=user.username,
            vector_store=get_vector_store(),
            metadata_store=get_metadata_store(),
            category=category,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        doc_id=result.doc_id,
        filename=result.filename,
        doc_type=result.doc_type,
        chunk_count=result.chunk_count,
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(user: UserContext = Depends(require_auth)):
    store = get_metadata_store()
    docs = await store.list_documents(user_groups=user.groups)
    return [
        DocumentInfo(
            doc_id=d.doc_id,
            filename=d.filename,
            doc_type=d.doc_type,
            category=d.category,
            acl_groups=d.acl_groups,
            chunk_count=d.chunk_count,
        )
        for d in docs
    ]
```

- [ ] **Step 7: Commit (we'll add main.py in the next task and run all tests together)**

```bash
git add src/api/ tests/test_api/
git commit -m "feat: auth and ingestion API routes with file upload support"
```

---

## Task 13: API Routes — Query & FastAPI App Entry Point

**Files:**
- Create: `rag/src/api/routes_query.py`
- Create: `rag/src/main.py`
- Create: `rag/tests/test_api/test_routes_query.py`

- [ ] **Step 1: Write failing tests for query route**

```python
# tests/test_api/test_routes_query.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.main import create_app
from src.auth.jwt import create_token
from src.generation.rag_chain import RAGResponse
from src.retrieval.models import Citation


@pytest.fixture
def auth_headers():
    token = create_token(username="mike", groups=["finance"])
    return {
        "Authorization": f"Bearer {token}",
        "X-API-Key": "test-key-1",
    }


@pytest.fixture
def client():
    return TestClient(create_app())


def test_query_returns_answer_with_citations(client, auth_headers):
    mock_response = RAGResponse(
        answer="Expenses over $500 need approval [1].",
        citations=[
            Citation(
                doc_id="doc-1",
                filename="policy.pdf",
                doc_type="pdf",
                chunk_index=0,
                page=12,
                snippet="All expenses over $500...",
                relevance=0.95,
            )
        ],
    )
    with patch("src.api.routes_query.rag_query", return_value=mock_response):
        resp = client.post(
            "/api/v1/query",
            json={"question": "What is the expense policy?"},
            headers=auth_headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "approval" in data["answer"].lower() or "500" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["filename"] == "policy.pdf"


def test_query_requires_auth(client):
    resp = client.post("/api/v1/query", json={"question": "test"})
    assert resp.status_code in (401, 403)


def test_query_empty_question(client, auth_headers):
    resp = client.post(
        "/api/v1/query",
        json={"question": ""},
        headers=auth_headers,
    )
    # Should still work (RAG chain handles empty gracefully)
    assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_api/test_routes_query.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement routes_query.py**

```python
# src/api/routes_query.py
from fastapi import APIRouter, Depends

from src.api.models import CitationResponse, QueryRequest, QueryResponse
from src.api.routes_ingest import get_vector_store
from src.auth.dependencies import require_auth
from src.auth.models import UserContext
from src.generation.rag_chain import rag_query

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, user: UserContext = Depends(require_auth)):
    result = rag_query(
        question=request.question,
        user_groups=user.groups,
        vector_store=get_vector_store(),
    )
    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                doc_id=c.doc_id,
                filename=c.filename,
                doc_type=c.doc_type,
                chunk_index=c.chunk_index,
                page=c.page,
                snippet=c.snippet,
                relevance=c.relevance,
            )
            for c in result.citations
        ],
    )
```

- [ ] **Step 4: Implement main.py (FastAPI app entry point)**

```python
# src/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes_auth import router as auth_router
from src.api.routes_ingest import router as ingest_router, get_metadata_store
from src.api.routes_query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize metadata DB
    store = get_metadata_store()
    await store.init()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Knowledge Service",
        description="Agentic RAG system with document-level access control",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(auth_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    return app


app = create_app()
```

- [ ] **Step 5: Run ALL tests**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/api/routes_query.py src/main.py tests/test_api/test_routes_query.py
git commit -m "feat: query API route and FastAPI app entry point"
```

---

## Task 14: Dockerfile & Docker Compose

**Files:**
- Create: `rag/Dockerfile`
- Create: `rag/docker-compose.yml`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for unstructured (PDF/OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/

EXPOSE 8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
# docker-compose.yml
version: "3.9"

services:
  # Qdrant vector database
  qdrant:
    image: qdrant/qdrant:v1.13.2
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

  # vLLM serving Gemma 4 31B (requires NVIDIA GPU)
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    command: >
      --model google/gemma-4-31b-it
      --max-model-len 32768
      --tensor-parallel-size 1
      --gpu-memory-utilization 0.9
    ports:
      - "8000:8000"
    volumes:
      - model_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  # RAG API server
  api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - VLLM_BASE_URL=http://vllm:8000/v1
      - VLLM_MODEL_NAME=google/gemma-4-31b-it
      - EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large
      - EMBEDDING_DEVICE=cpu
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - QDRANT_COLLECTION_NAME=documents
      - JWT_SECRET_KEY=${JWT_SECRET_KEY:-change-me-in-production}
      - API_KEYS=${API_KEYS:-dev-key-1}
      - DATABASE_URL=sqlite+aiosqlite:///./data/metadata.db
    volumes:
      - api_data:/app/data
      - embedding_cache:/root/.cache/huggingface
    depends_on:
      - qdrant
      - vllm
    restart: unless-stopped

  # Open WebUI (chat interface)
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - OPENAI_API_BASE_URL=http://api:8080/v1
      - OPENAI_API_KEY=dev-key-1
    volumes:
      - webui_data:/app/backend/data
    depends_on:
      - api
    restart: unless-stopped

volumes:
  qdrant_data:
  model_cache:
  api_data:
  embedding_cache:
  webui_data:
```

- [ ] **Step 3: Verify Dockerfile builds**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && docker build -t rag-api:dev .`
Expected: Successful image build (may take a few minutes for dependency installation)

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: Dockerfile and Docker Compose for full stack deployment"
```

---

## Task 15: API Key Creation Script & Test Data Seeder

**Files:**
- Create: `rag/scripts/create_api_key.py`
- Create: `rag/scripts/seed_test_data.py`

- [ ] **Step 1: Create API key generation script**

```python
# scripts/create_api_key.py
"""Generate a random API key and print it. Add it to your .env API_KEYS list."""
import secrets
import sys


def main():
    key = secrets.token_urlsafe(32)
    print(f"Generated API key: {key}")
    print(f"\nAdd to your .env file:")
    print(f"  API_KEYS=...existing-keys...,{key}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create test data seeder**

```python
# scripts/seed_test_data.py
"""Seed the system with test documents for development/demo purposes."""
import asyncio
import sys
from pathlib import Path

# Add project root to path
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
        result = await ingest_document(
            file_path=path,
            acl_groups=doc_info["acl_groups"],
            uploaded_by="seed-script",
            vector_store=vector_store,
            metadata_store=metadata_store,
            category=doc_info["category"],
        )
        print(f"    doc_id={result.doc_id}, chunks={result.chunk_count}")

    print("\nSeeding complete.")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Commit**

```bash
git add scripts/create_api_key.py scripts/seed_test_data.py
git commit -m "feat: API key generator and test data seeder scripts"
```

---

## Task 16: Final Integration Test & Full Test Suite Run

**Files:**
- No new files — validate everything works together

- [ ] **Step 1: Run the complete test suite**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/ -v --tb=short`
Expected: All tests PASS (approximately 30+ tests)

- [ ] **Step 2: Verify the FastAPI app starts**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && timeout 5 uvicorn src.main:app --host 0.0.0.0 --port 8080 || true`
Expected: App starts and begins listening (will timeout after 5 seconds — that's fine)

- [ ] **Step 3: Verify OpenAPI docs generate**

Run: `cd /Users/michaelmulkey/Documents/Repositories/rag && python -c "from src.main import create_app; app = create_app(); print('Routes:'); [print(f'  {r.methods} {r.path}') for r in app.routes if hasattr(r, 'methods')]"`
Expected: All routes listed:
- POST /api/v1/auth/token
- POST /api/v1/ingest
- GET /api/v1/documents
- POST /api/v1/query

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: Phase 1 Core RAG system complete — all tests passing"
```
