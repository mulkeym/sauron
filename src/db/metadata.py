from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.db.models import Base, DocumentRecord, Category, CategoryProposal


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

    async def add_category(self, name, description, acl_groups, routing_keywords):
        record = Category(name=name, description=description, acl_groups=acl_groups, routing_keywords=routing_keywords)
        async with self.session_factory() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
        return record

    async def get_category(self, name):
        async with self.session_factory() as session:
            result = await session.execute(select(Category).where(Category.name == name))
            return result.scalar_one_or_none()

    async def list_categories(self):
        async with self.session_factory() as session:
            result = await session.execute(select(Category))
            return list(result.scalars().all())

    async def add_proposal(self, proposed_name, proposed_description, proposed_acl_groups, proposed_keywords, proposed_by):
        record = CategoryProposal(proposed_name=proposed_name, proposed_description=proposed_description, proposed_acl_groups=proposed_acl_groups, proposed_keywords=proposed_keywords, proposed_by=proposed_by)
        async with self.session_factory() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
        return record

    async def list_proposals(self, status="pending"):
        async with self.session_factory() as session:
            result = await session.execute(select(CategoryProposal).where(CategoryProposal.status == status))
            return list(result.scalars().all())

    async def approve_proposal(self, proposal_id, approved_by):
        async with self.session_factory() as session:
            proposal = await session.get(CategoryProposal, proposal_id)
            if proposal:
                proposal.status = "approved"
                proposal.reviewed_by = approved_by
                cat = Category(name=proposal.proposed_name, description=proposal.proposed_description, acl_groups=proposal.proposed_acl_groups, routing_keywords=proposal.proposed_keywords)
                session.add(cat)
                await session.commit()

    async def reject_proposal(self, proposal_id, rejected_by):
        async with self.session_factory() as session:
            proposal = await session.get(CategoryProposal, proposal_id)
            if proposal:
                proposal.status = "rejected"
                proposal.reviewed_by = rejected_by
                await session.commit()
