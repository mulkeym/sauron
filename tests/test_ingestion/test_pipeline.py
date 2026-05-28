import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, call

from src.ingestion.pipeline import IngestResult, ingest_document
from src.knowledge.categorizer import CategorizationResult

FIXTURES = Path(__file__).parent.parent.parent / "test_fixtures"


@pytest.fixture
def mock_deps():
    mock_vector_store = MagicMock()
    mock_metadata_store = AsyncMock()
    mock_embed = MagicMock(return_value=[[0.1] * 1024])
    return mock_vector_store, mock_metadata_store, mock_embed


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


@pytest.mark.asyncio
async def test_ingest_auto_categorizes(mock_deps):
    vector_store, metadata_store, mock_embed = mock_deps
    metadata_store.list_categories = AsyncMock(return_value=[])
    metadata_store.add_proposal = AsyncMock()

    mock_cat_result = CategorizationResult(category="finance_policies", confidence=0.9, is_new=False)

    with patch("src.ingestion.pipeline.embed_texts", mock_embed):
        with patch("src.ingestion.pipeline.categorize_document", return_value=mock_cat_result):
            result = await ingest_document(
                file_path=FIXTURES / "sample.pdf", acl_groups=["finance"],
                uploaded_by="mike", vector_store=vector_store, metadata_store=metadata_store,
                auto_categorize=True,
            )
    assert result.doc_type == "pdf"
    metadata_store.add_document.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_extracts_entities(mock_deps):
    vector_store, metadata_store, mock_embed = mock_deps
    metadata_store.add_entity = AsyncMock(return_value=1)
    metadata_store.add_mention = AsyncMock()
    metadata_store.add_relationship = AsyncMock()

    from src.knowledge.extractor import ExtractionResult
    mock_extraction = ExtractionResult(
        entities=[{"name": "Mike", "type": "person"}, {"name": "Policy 4.2", "type": "policy"}],
        relationships=[{"source": "Policy 4.2", "target": "expense reporting", "type": "governs"}],
        sections=[],
    )

    with patch("src.ingestion.pipeline.embed_texts", mock_embed):
        with patch("src.ingestion.pipeline.extract_entities", return_value=mock_extraction):
            result = await ingest_document(
                file_path=FIXTURES / "sample.pdf", acl_groups=["finance"],
                uploaded_by="mike", vector_store=vector_store, metadata_store=metadata_store,
            )
    assert result.chunk_count > 0
    metadata_store.add_entity.assert_called()
    metadata_store.add_mention.assert_called()


@pytest.mark.asyncio
async def test_ingest_document_spreadsheet_dedups_clean_sheet(tmp_path, monkeypatch):
    """A clean spreadsheet that is structured-ingested emits NO full-text tier
    chunks (only its DuckDB rows + table_row narratives); messy sheets would
    still produce structure-aware text chunks."""
    import openpyxl
    from src.ingestion import pipeline

    p = tmp_path / "pay.xlsx"
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    clean = wb.create_sheet("Pay")
    for r in [["locality", "grade", "salary"], ["Tampa", "GS-12", 86415],
              ["Boston", "GS-12", 92000], ["Denver", "GS-13", 99000]]:
        clean.append(r)
    wb.save(p)

    captured = {"tiers": []}

    class FakeVS:
        def upsert(self, texts, vectors, metadatas):
            captured["tiers"] += [m.chunk_size_tier for m in metadatas]

    class FakeMS:
        async def add_document(self, **k): pass
        async def get_category(self, name): return None
        async def add_category(self, **k): pass
        async def save_schema(self, s): pass

    # Stub everything external: embeddings (both call sites), the LLM summary/
    # profiler, the LightRAG insert, and the schema registry.
    monkeypatch.setattr(pipeline, "embed_texts", lambda texts, *a, **k: [[0.0] for _ in texts])
    monkeypatch.setattr("src.ingestion.tabular_ingest.embed_texts",
                        lambda texts, *a, **k: [[0.0] for _ in texts])
    monkeypatch.setattr("src.generation.llm_client.generate", lambda **k: "")

    async def _coro(*a, **k):
        return None
    monkeypatch.setattr("src.knowledge.graph_rag.insert_document", lambda *a, **k: _coro())
    monkeypatch.setattr("src.api.routes_ingest.get_schema_registry",
                        lambda: type("R", (), {"register": lambda self, s: None,
                                               "remove": lambda self, *a: None})())

    await pipeline.ingest_document(
        str(p), acl_groups=["g1"], uploaded_by="t", vector_store=FakeVS(),
        metadata_store=FakeMS(), category="cat",
    )
    # The clean "Pay" sheet was de-duped: no small/medium/large/xlarge text tiers,
    # only table_row narratives.
    assert "table_row" in captured["tiers"]
    assert not ({"small", "medium", "large", "xlarge"} & set(captured["tiers"]))


def _stub_pipeline(monkeypatch, pipeline, kg_calls):
    """Stub the external deps shared by the KG-gating tests, recording every
    LightRAG insert_document call into kg_calls."""
    async def _coro(*a, **k):
        return None

    def _fake_insert(*a, **k):
        kg_calls.append((a, k))
        return _coro()

    monkeypatch.setattr(pipeline, "embed_texts", lambda texts, *a, **k: [[0.0] for _ in texts])
    monkeypatch.setattr("src.ingestion.tabular_ingest.embed_texts",
                        lambda texts, *a, **k: [[0.0] for _ in texts])
    monkeypatch.setattr("src.generation.llm_client.generate", lambda **k: "")
    monkeypatch.setattr("src.knowledge.graph_rag.insert_document", _fake_insert)
    monkeypatch.setattr("src.api.routes_ingest.get_schema_registry",
                        lambda: type("R", (), {"register": lambda self, s: None,
                                               "remove": lambda self, *a: None})())


class _GateVS:
    def upsert(self, texts, vectors, metadatas): pass


class _GateMS:
    async def add_document(self, **k): pass
    async def get_category(self, name): return None
    async def add_category(self, **k): pass
    async def save_schema(self, s): pass


@pytest.mark.asyncio
async def test_ingest_spreadsheet_skips_knowledge_graph(tmp_path, monkeypatch):
    """Spreadsheets are fully covered by the structured/tabular store, so KG
    extraction (costly + near-empty for numeric tables) is skipped."""
    import openpyxl
    from src.ingestion import pipeline

    p = tmp_path / "pay.xlsx"
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    s = wb.create_sheet("Pay")
    for r in [["locality", "grade", "salary"], ["Tampa", "GS-12", 86415]]:
        s.append(r)
    wb.save(p)

    kg_calls = []
    _stub_pipeline(monkeypatch, pipeline, kg_calls)
    await pipeline.ingest_document(str(p), acl_groups=["g1"], uploaded_by="t",
                                   vector_store=_GateVS(), metadata_store=_GateMS(), category="cat")
    assert kg_calls == []   # KG skipped for spreadsheet


@pytest.mark.asyncio
async def test_ingest_plaintext_still_builds_knowledge_graph(tmp_path, monkeypatch):
    """Regression guard: the spreadsheet gate must NOT suppress KG for prose docs."""
    from src.ingestion import pipeline

    p = tmp_path / "memo.txt"
    p.write_text("Acme Corp signed a contract with Globex in 2026.")

    kg_calls = []
    _stub_pipeline(monkeypatch, pipeline, kg_calls)
    await pipeline.ingest_document(str(p), acl_groups=["g1"], uploaded_by="t",
                                   vector_store=_GateVS(), metadata_store=_GateMS(), category="cat")
    assert len(kg_calls) == 1   # KG still runs for non-spreadsheet
