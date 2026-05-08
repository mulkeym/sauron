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


@pytest.mark.asyncio
async def test_get_nonexistent_document(store):
    doc = await store.get_document("does-not-exist")
    assert doc is None


@pytest.mark.asyncio
async def test_list_documents(store):
    await store.add_document(
        doc_id="d1", filename="a.pdf", doc_type="pdf", acl_groups=["finance"], chunk_count=3, uploaded_by="mike"
    )
    await store.add_document(
        doc_id="d2", filename="b.docx", doc_type="docx", acl_groups=["it"], chunk_count=2, uploaded_by="bob"
    )
    docs = await store.list_documents()
    assert len(docs) == 2


@pytest.mark.asyncio
async def test_list_documents_filtered_by_groups(store):
    await store.add_document(
        doc_id="d1", filename="a.pdf", doc_type="pdf", acl_groups=["finance"], chunk_count=3, uploaded_by="mike"
    )
    await store.add_document(
        doc_id="d2", filename="b.docx", doc_type="docx", acl_groups=["it"], chunk_count=2, uploaded_by="bob"
    )
    docs = await store.list_documents(user_groups=["finance"])
    assert len(docs) == 1
    assert docs[0].filename == "a.pdf"


@pytest.mark.asyncio
async def test_delete_document(store):
    await store.add_document(
        doc_id="d1", filename="a.pdf", doc_type="pdf", acl_groups=["finance"], chunk_count=3, uploaded_by="mike"
    )
    await store.delete_document("d1")
    doc = await store.get_document("d1")
    assert doc is None
