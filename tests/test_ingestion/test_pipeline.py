import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from src.ingestion.pipeline import IngestResult, ingest_document

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
