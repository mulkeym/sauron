from fastmcp import FastMCP
from src.db.metadata import MetadataStore
from src.db.schema_registry import SchemaRegistry
from src.mcp.agent_registry import AgentRegistry
from src.mcp.jobs import JobStore
from src.mcp.tools_high import ask, summarize_topic, compare, summarize_documents
from src.mcp.tools_low import search_documents, query_database, lookup_document, search_meetings, list_sources, list_documents_in_category, search_knowledge_graph
from src.mcp.resources import get_document_resource, get_category_resource, get_schema_resource
from src.retrieval.vector_store import VectorStore


def create_mcp_server(
    vector_store: VectorStore,
    schema_registry: SchemaRegistry,
    metadata_store: MetadataStore,
    agent_registry: AgentRegistry,
) -> FastMCP:
    mcp = FastMCP("rag-knowledge-service")
    job_store = JobStore()

    @mcp.tool()
    async def tool_ask(question: str, depth: str = "thorough", context: str = "") -> dict:
        """Ask a question about document CONTENT and get a cited answer. Use this for questions like 'what does the policy say?' or 'what are the budget numbers?'. Do NOT use this for listing files, reading specific files, or finding what documents exist — use tool_list_documents and tool_lookup_document for those tasks instead."""
        return await ask(
            question=question,
            user_groups=["ALL"],
            vector_store=vector_store,
            schema_registry=schema_registry,
            metadata_store=metadata_store,
            depth=depth,
            context=context or None,
        )

    @mcp.tool()
    async def tool_summarize_topic(topic: str, format: str = "brief") -> dict:
        """Summarize a specific topic by searching across all documents and generating a summary with source references."""
        return await summarize_topic(
            topic=topic,
            user_groups=["ALL"],
            vector_store=vector_store,
            schema_registry=schema_registry,
            metadata_store=metadata_store,
            format=format,
        )

    @mcp.tool()
    async def tool_compare(item_a: str, item_b: str) -> dict:
        """Compare and contrast two items, policies, or topics by searching the documents for both and listing differences."""
        return await compare(
            item_a=item_a,
            item_b=item_b,
            user_groups=["ALL"],
            vector_store=vector_store,
            schema_registry=schema_registry,
            metadata_store=metadata_store,
        )

    @mcp.tool()
    def tool_search_documents(query: str, doc_type: str = "", top_k: int = 10) -> list[dict]:
        """Search for relevant document snippets matching a query. Returns matching text chunks with filenames, doc_ids, and relevance scores. Use this to find which documents mention a topic. Each result includes a doc_id that can be used with tool_lookup_document to read the full document. Optional doc_type filter: pdf, docx, xlsx, transcript."""
        return search_documents(
            query=query,
            user_groups=["ALL"],
            vector_store=vector_store,
            doc_type=doc_type or None,
            top_k=top_k,
        )

    @mcp.tool()
    async def tool_query_database(question: str) -> dict:
        """Query a structured database using natural language. Converts your question to SQL, executes it, and returns the results. Use for questions about numbers, financial data, or anything stored in tables. If no database is configured, automatically searches documents instead."""
        return await query_database(
            question=question,
            user_groups=["ALL"],
            schema_registry=schema_registry,
            vector_store=vector_store,
            metadata_store=metadata_store,
        )

    @mcp.tool()
    def tool_lookup_document(doc_id: str) -> dict:
        """Read the full content of a document. Accepts either a doc_id (UUID) or a filename. Use this when the user asks to read, view, summarize, or display a specific file. You can pass the filename directly (e.g., 'sample.pdf') or a doc_id from tool_list_documents."""
        return lookup_document(
            doc_id=doc_id,
            user_groups=["ALL"],
            vector_store=vector_store,
        )

    @mcp.tool()
    def tool_search_meetings(topic: str = "", speaker: str = "", type_filter: str = "") -> list[dict]:
        """Search meeting transcripts. Filter by speaker name, topic, or utterance type (question, statement, action_item). Use this to find what someone said in meetings or to find specific discussions."""
        return search_meetings(
            user_groups=["ALL"],
            vector_store=vector_store,
            topic=topic or None,
            speaker=speaker or None,
            type_filter=type_filter or None,
        )

    @mcp.tool()
    def tool_list_sources() -> list[dict]:
        """List all available document categories and their document counts. Shows what knowledge sources exist in the system (e.g., finance_policies, it_runbooks, meeting_notes)."""
        return list_sources(user_groups=["ALL"], metadata_store=metadata_store)

    @mcp.tool()
    def tool_summarize_documents(category: str = "") -> dict:
        """Read and summarize every document in a category. Returns a list of filenames with a 2-3 sentence summary of each. Use this when the user asks to summarize multiple documents, summarize a category, or wants an overview of what's in a group of files. For a single file, use tool_lookup_document instead."""
        return summarize_documents(
            category=category or "uncategorized",
            user_groups=["ALL"],
            vector_store=vector_store,
            metadata_store=metadata_store,
        )

    @mcp.tool()
    def tool_list_documents(category: str = "") -> list[dict]:
        """List all documents with filenames, doc_ids, types, and categories. Use this FIRST when the user asks about what files exist, what's in a category, or wants to see uncategorized documents. Filter by category name (e.g., 'meeting_notes', 'finance_policies', 'uncategorized'). To read a file's content after listing, pass its doc_id or filename to tool_lookup_document."""
        if category:
            return list_documents_in_category(category=category, user_groups=["ALL"], metadata_store=metadata_store)
        # Return all documents
        import asyncio
        try:
            docs = asyncio.run(metadata_store.list_documents(None))
        except RuntimeError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                docs = executor.submit(asyncio.run, metadata_store.list_documents(None)).result()
        return [{"doc_id": d.doc_id, "filename": d.filename, "doc_type": d.doc_type, "category": d.category or "uncategorized", "chunk_count": d.chunk_count, "uploaded_by": d.uploaded_by} for d in docs]

    @mcp.tool()
    async def tool_search_knowledge_graph(query: str, entity_type: str = "") -> dict:
        """Search the knowledge graph for an entity (person, policy, project, organization, etc.) and find all related entities, relationships, and source documents. Use this to understand how concepts are connected across documents. Example: 'TOEE 26' returns related projects, people, and organizations."""
        return await search_knowledge_graph(query=query, metadata_store=metadata_store, entity_type=entity_type or None)

    @mcp.tool()
    def tool_get_result(job_id: str) -> dict:
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
        return await get_document_resource(doc_id, user_groups=["ALL"], metadata_store=metadata_store)

    @mcp.resource("category://{category_name}")
    def resource_category(category_name: str) -> dict:
        """Get category details and document list."""
        return get_category_resource(category_name, user_groups=["ALL"], metadata_store=metadata_store)

    @mcp.resource("schema://{database_name}")
    def resource_schema(database_name: str) -> dict:
        """Get database schema details."""
        return get_schema_resource(database_name, user_groups=["ALL"], schema_registry=schema_registry)

    return mcp
