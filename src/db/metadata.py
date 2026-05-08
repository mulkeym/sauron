from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

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
        doc_id,
        filename,
        doc_type,
        acl_groups,
        chunk_count,
        uploaded_by,
        category="",
    ):
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

    async def get_document(self, doc_id):
        async with self.session_factory() as session:
            return await session.get(DocumentRecord, doc_id)

    async def list_documents(self, user_groups=None):
        async with self.session_factory() as session:
            result = await session.execute(select(DocumentRecord))
            docs = list(result.scalars().all())
        if user_groups is not None:
            docs = [d for d in docs if any(g in d.acl_groups for g in user_groups)]
        return docs

    async def delete_document(self, doc_id):
        async with self.session_factory() as session:
            await session.execute(delete(DocumentRecord).where(DocumentRecord.doc_id == doc_id))
            await session.commit()
