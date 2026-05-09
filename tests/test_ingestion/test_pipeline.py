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
