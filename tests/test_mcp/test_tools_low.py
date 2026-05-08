import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.retrieval.models import RetrievedChunk, ChunkMetadata


def _make_chunk(text, doc_type="pdf", filename="test.pdf", speaker=None, utterance_type=None):
    return RetrievedChunk(
        text=text,
        score=0.9,
        metadata=ChunkMetadata(
            doc_id="d1",
            filename=filename,
            doc_type=doc_type,
            chunk_index=0,
            start_char=0,
            acl_groups=["finance"],
            speaker=speaker,
            utterance_type=utterance_type,
        ),
    )


def test_search_documents():
    from src.mcp.tools_low import search_documents

    mock_store = MagicMock()
    mock_store.search.return_value = [_make_chunk("Some content")]
    with patch("src.mcp.tools_low.embed_query", return_value=[0.1] * 1024):
        result = search_documents(query="test", user_groups=["finance"], vector_store=mock_store)
    assert len(result) == 1
    assert result[0]["text"] == "Some content"
    assert result[0]["source"] == "test.pdf"


def test_search_documents_with_doc_type_filter():
    from src.mcp.tools_low import search_documents

    chunk_pdf = _make_chunk("PDF content", doc_type="pdf")
    chunk_docx = _make_chunk("Word content", doc_type="docx", filename="test.docx")
    mock_store = MagicMock()
    mock_store.search.return_value = [chunk_pdf, chunk_docx]
    with patch("src.mcp.tools_low.embed_query", return_value=[0.1] * 1024):
        result = search_documents(
            query="test", user_groups=["finance"], vector_store=mock_store, doc_type="pdf"
        )
    assert len(result) == 1
    assert result[0]["source"] == "test.pdf"


@pytest.mark.asyncio
async def test_query_database():
    from src.mcp.tools_low import query_database
    from src.db.schema_registry import SchemaRegistry, TableSchema, ColumnSchema

    registry = SchemaRegistry()
    registry.register(
        TableSchema(
            database="finance_db",
            table="results",
            columns=[ColumnSchema(name="revenue", dtype="numeric", description="USD")],
            description="Results",
            acl_groups=["finance"],
        )
    )
    with patch("src.mcp.tools_low.generate", return_value="SELECT revenue FROM results WHERE quarter='Q3'"):
        with patch(
            "src.mcp.tools_low.execute_sql",
            new_callable=AsyncMock,
            return_value=[{"revenue": 1500000}],
        ):
            result = await query_database(
                question="Q3 revenue?", user_groups=["finance"], schema_registry=registry
            )
    assert result["results"] == [{"revenue": 1500000}]


def test_search_meetings():
    from src.mcp.tools_low import search_meetings

    chunks = [
        _make_chunk(
            "Mike: Are we on track?",
            doc_type="transcript",
            filename="standup.txt",
            speaker="Mike",
            utterance_type="question",
        ),
        _make_chunk(
            "Sarah: Yes",
            doc_type="transcript",
            filename="standup.txt",
            speaker="Sarah",
            utterance_type="statement",
        ),
    ]
    mock_store = MagicMock()
    mock_store.search.return_value = chunks
    with patch("src.mcp.tools_low.embed_query", return_value=[0.1] * 1024):
        result = search_meetings(
            user_groups=["engineering"],
            vector_store=mock_store,
            speaker="Mike",
            type_filter="question",
        )
    assert len(result) == 1
    assert result[0]["speaker"] == "Mike"


def test_list_sources():
    from src.mcp.tools_low import list_sources

    mock_metadata = MagicMock()
    mock_doc = MagicMock()
    mock_doc.category = "finance_policies"
    mock_doc.doc_type = "pdf"
    # Make list_documents return a coroutine that returns the list
    mock_metadata.list_documents = AsyncMock(return_value=[mock_doc, mock_doc, mock_doc])
    result = list_sources(user_groups=["finance"], metadata_store=mock_metadata)
    assert len(result) >= 1
