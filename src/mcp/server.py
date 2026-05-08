from fastmcp import FastMCP
from src.db.metadata import MetadataStore
from src.db.schema_registry import SchemaRegistry
from src.mcp.agent_registry import AgentRegistry
from src.mcp.jobs import JobStore
from src.mcp.tools_high import ask, summarize_topic, compare
from src.mcp.tools_low import search_documents, query_database, lookup_document, search_meetings, list_sources
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
        """Ask a question and get a complete cited answer from the knowledge base."""
        return await ask(
            question=question,
            user_groups=["ALL"],
            vector_store=vector_store,
            schema_registry=schema_registry,
            depth=depth,
            context=context or None,
        )

    @mcp.tool()
    async def tool_summarize_topic(topic: str, format: str = "brief") -> dict:
        """Summarize a topic from the knowledge base."""
        return await summarize_topic(
            topic=topic,
            user_groups=["ALL"],
            vector_store=vector_store,
            schema_registry=schema_registry,
            format=format,
        )

    @mcp.tool()
    async def tool_compare(item_a: str, item_b: str) -> dict:
        """Compare two items using the knowledge base."""
        return await compare(
            item_a=item_a,
            item_b=item_b,
            user_groups=["ALL"],
            vector_store=vector_store,
            schema_registry=schema_registry,
        )

    @mcp.tool()
    def tool_search_documents(query: str, doc_type: str = "", top_k: int = 10) -> list[dict]:
        """Search documents by semantic similarity with optional type filter."""
        return search_documents(
            query=query,
            user_groups=["ALL"],
            vector_store=vector_store,
            doc_type=doc_type or None,
            top_k=top_k,
        )

    @mcp.tool()
    async def tool_query_database(question: str) -> dict:
        """Query a registered database using natural language (text-to-SQL)."""
        return await query_database(
            question=question,
            user_groups=["ALL"],
            schema_registry=schema_registry,
        )

    @mcp.tool()
    def tool_lookup_document(doc_id: str) -> dict:
        """Retrieve a specific document by ID."""
        return lookup_document(
            doc_id=doc_id,
            user_groups=["ALL"],
            vector_store=vector_store,
        )

    @mcp.tool()
    def tool_search_meetings(topic: str = "", speaker: str = "", type_filter: str = "") -> list[dict]:
        """Search meeting transcripts with optional speaker and type filters."""
        return search_meetings(
            user_groups=["ALL"],
            vector_store=vector_store,
            topic=topic or None,
            speaker=speaker or None,
            type_filter=type_filter or None,
        )

    @mcp.tool()
    def tool_list_sources() -> list[dict]:
        """List available knowledge sources and their document counts."""
        return list_sources(user_groups=["ALL"], metadata_store=metadata_store)

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
