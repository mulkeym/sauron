import asyncio

from fastmcp import FastMCP
from src.db.metadata import MetadataStore
from src.db.schema_registry import SchemaRegistry
from src.mcp.agent_registry import AgentRegistry
from src.mcp.jobs import JobStore
from src.mcp.tools_high import ask, summarize_topic, compare, summarize_documents
from src.mcp.tools_low import search_documents, query_database, lookup_document, search_meetings, list_sources, list_documents_in_category, search_knowledge_graph
from src.mcp.resources import get_document_resource, get_category_resource, get_schema_resource
from src.mcp.activity_wrap import run_logged_mcp_tool
from src.mcp.auth import current_mcp_context
from src.retrieval.vector_store import VectorStore
from src.config import settings


def create_mcp_server(
    vector_store: VectorStore,
    schema_registry: SchemaRegistry,
    metadata_store: MetadataStore,
    agent_registry: AgentRegistry,
) -> FastMCP:
    mcp = FastMCP(settings.mcp_server_name)
    job_store = JobStore()

    def user_groups() -> list[str]:
        """Resolve ACL groups from the authenticated request; never default to ALL."""
        return current_mcp_context().groups

    @mcp.tool()
    async def tool_ask(question: str, depth: str = "thorough", context: str = "") -> dict:
        """THIS IS THE PRIMARY TOOL — use it for ANY question about document content, contracts, policies, people, companies, awards, or facts. It searches all documents, enriches with knowledge graph data, and generates a comprehensive cited answer. Use this FIRST before trying other tools. Only use tool_list_documents or tool_lookup_document for browsing/reading specific files."""
        async def _run():
            return await ask(
                question=question,
                user_groups=user_groups(),
                vector_store=vector_store,
                schema_registry=schema_registry,
                metadata_store=metadata_store,
                depth=depth,
                context=context or None,
            )
        return await run_logged_mcp_tool(tool="ask", query_text=question, fn=_run)

    @mcp.tool()
    async def tool_summarize_topic(topic: str, format: str = "brief") -> dict:
        """Summarize a specific topic by searching across all documents and generating a summary with source references."""
        async def _run():
            return await summarize_topic(
                topic=topic,
                user_groups=user_groups(),
                vector_store=vector_store,
                schema_registry=schema_registry,
                metadata_store=metadata_store,
                format=format,
            )
        return await run_logged_mcp_tool(tool="summarize_topic", query_text=topic, fn=_run)

    @mcp.tool()
    async def tool_compare(item_a: str, item_b: str) -> dict:
        """Compare and contrast two items, policies, or topics by searching the documents for both and listing differences."""
        async def _run():
            return await compare(
                item_a=item_a,
                item_b=item_b,
                user_groups=user_groups(),
                vector_store=vector_store,
                schema_registry=schema_registry,
                metadata_store=metadata_store,
            )
        return await run_logged_mcp_tool(
            tool="compare", query_text=f"{item_a} vs {item_b}", fn=_run,
        )

    @mcp.tool()
    async def tool_search_documents(query: str, doc_type: str = "", top_k: int = 10) -> list[dict]:
        """Low-level search returning raw text snippets. For ANSWERING questions, use tool_ask instead — it provides better results with knowledge graph enrichment and cited answers. Only use this tool when you need raw document snippets for your own analysis, or to find doc_ids for tool_lookup_document."""
        async def _run():
            return await asyncio.to_thread(
                search_documents, query=query, user_groups=user_groups(),
                vector_store=vector_store, doc_type=doc_type or None, top_k=top_k,
            )
        return await run_logged_mcp_tool(tool="search_documents", query_text=query, fn=_run)

    @mcp.tool()
    async def tool_query_database(question: str) -> dict:
        """Query a structured SQL database only. For ANSWERING questions about document content, contracts, policies, or facts, use tool_ask instead. Only use this when you specifically need to run a SQL query against a registered database."""
        async def _run():
            return await query_database(
                question=question,
                user_groups=user_groups(),
                schema_registry=schema_registry,
                vector_store=vector_store,
                metadata_store=metadata_store,
            )
        return await run_logged_mcp_tool(
            tool="query_database", query_text=question, fn=_run, default_strategy="structured",
        )

    @mcp.tool()
    async def tool_lookup_document(doc_id: str) -> dict:
        """Read the full content of a document. Accepts either a doc_id (UUID) or a filename. Use this when the user asks to read, view, summarize, or display a specific file. You can pass the filename directly (e.g., 'sample.pdf') or a doc_id from tool_list_documents."""
        async def _run():
            return await asyncio.to_thread(
                lookup_document, doc_id=doc_id, user_groups=user_groups(),
                vector_store=vector_store,
            )
        return await run_logged_mcp_tool(tool="lookup_document", query_text=doc_id, fn=_run)

    @mcp.tool()
    async def tool_search_meetings(topic: str = "", speaker: str = "", type_filter: str = "") -> list[dict]:
        """Search meeting transcripts. Filter by speaker name, topic, or utterance type (question, statement, action_item). Use this to find what someone said in meetings or to find specific discussions."""
        q = " / ".join(p for p in (topic, speaker, type_filter) if p)
        async def _run():
            return await asyncio.to_thread(
                search_meetings, user_groups=user_groups(),
                vector_store=vector_store, topic=topic or None,
                speaker=speaker or None, type_filter=type_filter or None,
            )
        return await run_logged_mcp_tool(tool="search_meetings", query_text=q, fn=_run)

    @mcp.tool()
    async def tool_list_sources() -> list[dict]:
        """List all available document categories and their document counts. Shows what knowledge sources exist in the system (e.g., finance_policies, it_runbooks, meeting_notes)."""
        async def _run():
            return await asyncio.to_thread(
                list_sources, user_groups=user_groups(), metadata_store=metadata_store,
            )
        return await run_logged_mcp_tool(tool="list_sources", query_text="", fn=_run)

    @mcp.tool()
    async def tool_summarize_documents(category: str = "") -> dict:
        """Read and summarize every document in a category. Returns a list of filenames with a 2-3 sentence summary of each. Use this when the user asks to summarize multiple documents, summarize a category, or wants an overview of what's in a group of files. For a single file, use tool_lookup_document instead."""
        async def _run():
            return await asyncio.to_thread(
                summarize_documents, category=category or "uncategorized",
                user_groups=user_groups(), vector_store=vector_store,
                metadata_store=metadata_store,
            )
        return await run_logged_mcp_tool(
            tool="summarize_documents",
            query_text=category,
            fn=_run,
        )

    @mcp.tool()
    async def tool_list_documents(category: str = "") -> list[dict]:
        """List all documents with filenames, doc_ids, types, and categories. Use this FIRST when the user asks about what files exist, what's in a category, or wants to see uncategorized documents. Filter by category name (e.g., 'meeting_notes', 'finance_policies', 'uncategorized'). To read a file's content after listing, pass its doc_id or filename to tool_lookup_document."""
        async def _run():
            groups = user_groups()
            if category:
                return list_documents_in_category(category=category, user_groups=groups, metadata_store=metadata_store)
            docs = await metadata_store.list_documents(None if "ALL" in groups else groups)
            return [{"doc_id": d.doc_id, "filename": d.filename, "doc_type": d.doc_type, "category": d.category or "uncategorized", "chunk_count": d.chunk_count, "uploaded_by": d.uploaded_by} for d in docs]
        return await run_logged_mcp_tool(tool="list_documents", query_text=category, fn=_run)

    @mcp.tool()
    async def tool_search_knowledge_graph(query: str, entity_type: str = "") -> dict:
        """Search the knowledge graph for an entity (person, policy, project, organization, etc.) and find all related entities, relationships, and source documents. Use this to understand how concepts are connected across documents. Example: 'TOEE 26' returns related projects, people, and organizations."""
        async def _run():
            return await search_knowledge_graph(
                query=query,
                user_groups=user_groups(),
                metadata_store=metadata_store,
                entity_type=entity_type or None,
            )
        return await run_logged_mcp_tool(tool="search_knowledge_graph", query_text=query, fn=_run)

    @mcp.tool()
    async def tool_get_result(job_id: str) -> dict:
        """Check the status of an async job."""
        job = job_store.get(job_id)
        if job is None:
            return {"error": "Job not found"}
        return {
            "job_id": job_id,
            "status": job["status"],
            "result": job.get("result"),
            "error": job.get("error"),
        }

    @mcp.resource("document://{doc_id}")
    async def resource_document(doc_id: str) -> dict:
        """Get document metadata by ID."""
        return await get_document_resource(doc_id, user_groups=user_groups(), metadata_store=metadata_store)

    @mcp.resource("category://{category_name}")
    def resource_category(category_name: str) -> dict:
        """Get category details and document list."""
        return get_category_resource(category_name, user_groups=user_groups(), metadata_store=metadata_store)

    @mcp.resource("schema://{database_name}")
    def resource_schema(database_name: str) -> dict:
        """Get database schema details."""
        return get_schema_resource(database_name, user_groups=user_groups(), schema_registry=schema_registry)

    return mcp
