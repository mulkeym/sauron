from __future__ import annotations
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState, QueryType
from src.agent.classifier import classify_query
from src.agent.synthesizer import synthesize_answer
from src.agent.strategies.lookup import retrieve_lookup
from src.agent.strategies.sweep import retrieve_sweep
from src.agent.strategies.analytical import retrieve_analytical
from src.agent.strategies.cross_reference import retrieve_cross_reference
from src.db.metadata import MetadataStore
from src.db.schema_registry import SchemaRegistry
from src.generation.rag_chain import RAGResponse
from src.retrieval.models import RetrievedChunk, ChunkMetadata
from src.retrieval.vector_store import VectorStore


def create_agent_graph(vector_store: VectorStore, schema_registry: SchemaRegistry, metadata_store: MetadataStore | None = None):
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_query)

    async def retrieve(state: AgentState) -> dict:
        import logging
        retrieve_logger = logging.getLogger("retrieval")
        query_type = state.get("query_type", QueryType.LOOKUP)
        attempts = state.get("retrieval_attempts", 0)

        # On retry, use reformulated query if available
        if attempts > 0 and state.get("reformulated_query"):
            retry_state = dict(state)
            retry_state["question"] = state["reformulated_query"]
        else:
            retry_state = state

        if query_type == QueryType.LOOKUP:
            result = retrieve_lookup(retry_state, vector_store=vector_store)
        elif query_type == QueryType.SWEEP:
            result = await retrieve_sweep(retry_state, vector_store=vector_store)
        elif query_type == QueryType.ANALYTICAL:
            result = await retrieve_analytical(retry_state, vector_store=vector_store, schema_registry=schema_registry)
        elif query_type == QueryType.CROSS_REFERENCE:
            result = await retrieve_cross_reference(retry_state, vector_store=vector_store, schema_registry=schema_registry)
        elif query_type == QueryType.TEMPORAL:
            result = retrieve_lookup(retry_state, vector_store=vector_store)
        else:
            result = retrieve_lookup(retry_state, vector_store=vector_store)

        # Sub-task decomposition: run additional searches for each sub-task IN PARALLEL
        sub_tasks = state.get("sub_tasks", [])
        unique_tasks = [t for t in sub_tasks if t != state["question"]] if sub_tasks else []
        if unique_tasks:
            from src.ingestion.embedder import embed_query
            import asyncio

            async def search_subtask(task):
                task_vector = await asyncio.to_thread(embed_query, task)
                return vector_store.hybrid_search(
                    vector=task_vector, text_query=task,
                    user_groups=state.get("user_groups", ["ALL"]),
                    top_k=10, tier="small",
                )

            all_subtask_results = await asyncio.gather(*[search_subtask(t) for t in unique_tasks])

            existing_keys = {(c.metadata.doc_id, c.metadata.chunk_index)
                            for c in result.get("retrieved_chunks", [])}
            added = 0
            for task_chunks in all_subtask_results:
                for c in task_chunks:
                    key = (c.metadata.doc_id, c.metadata.chunk_index)
                    if key not in existing_keys:
                        result.setdefault("retrieved_chunks", []).append(c)
                        existing_keys.add(key)
                        added += 1
            if added:
                retrieve_logger.info(f"Sub-task decomposition added {added} chunks from {len(unique_tasks)} sub-tasks (parallel)")

        # On retry, merge new chunks with existing (don't lose previous results)
        if attempts > 0:
            existing = state.get("retrieved_chunks", [])
            new_chunks = result.get("retrieved_chunks", [])
            seen = {(c.metadata.doc_id, c.metadata.chunk_index) for c in existing}
            for c in new_chunks:
                if (c.metadata.doc_id, c.metadata.chunk_index) not in seen:
                    existing.append(c)
                    seen.add((c.metadata.doc_id, c.metadata.chunk_index))
            result["retrieved_chunks"] = existing

        return result

    graph.add_node("retrieve", retrieve)

    # Enrich retrieved context with LightRAG knowledge graph
    async def enrich_with_graph(state: AgentState) -> dict:
        if state.get("skip_graph"):
            return {}
        # ACL safety: graph has no per-document ACL filtering, so only
        # enrich for users with ALL access to prevent data leakage
        user_groups = state.get("user_groups", [])
        if "ALL" not in user_groups:
            return {}
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return {}

        question = state.get("question", "")
        try:
            import logging
            from src.knowledge.graph_rag import query_graph, is_graph_populated

            kg_logger = logging.getLogger("knowledge_graph")

            # Skip if graph has no data
            if not await is_graph_populated():
                return {}

            # Query LightRAG for knowledge graph context
            result = await query_graph(question, mode="mix")
            kg_context = result.get("context", "")

            if not kg_context or len(kg_context.strip()) < 20:
                return {}

            kg_logger.info(f"LightRAG enrichment: {len(kg_context)} chars of graph context")

            # Add as a synthetic chunk
            kg_chunk = RetrievedChunk(
                text=f"Knowledge Graph Context:\n{kg_context}",
                score=0.5,
                metadata=ChunkMetadata(
                    doc_id="knowledge-graph", filename="knowledge_graph",
                    doc_type="graph", chunk_index=0, start_char=0, acl_groups=["ALL"],
                ),
            )
            return {"retrieved_chunks": chunks + [kg_chunk]}
        except Exception as e:
            import logging
            logging.getLogger("knowledge_graph").warning(f"Graph enrichment failed: {e}")
            return {}

    graph.add_node("enrich", enrich_with_graph)
    graph.add_node("synthesize", synthesize_answer)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "enrich")
    graph.add_edge("enrich", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


async def run_agent(question: str, user_groups: list[str], vector_store: VectorStore, schema_registry: SchemaRegistry, metadata_store: MetadataStore | None = None) -> RAGResponse:
    graph = create_agent_graph(vector_store=vector_store, schema_registry=schema_registry, metadata_store=metadata_store)
    initial_state = AgentState(
        question=question, original_question=question, user_groups=user_groups,
        query_type=None, sub_tasks=[], retrieved_chunks=[], sql_results=[],
        retrieval_attempts=0, needs_reretrieval=False, reformulated_query="",
        answer="", citations=[], warnings=[],
    )
    result = await graph.ainvoke(initial_state)
    return RAGResponse(
        answer=result.get("answer", "I could not find any relevant information."),
        citations=result.get("citations", []),
    )


@dataclass
class AgentTrace:
    steps: list = field(default_factory=list)
    total_time: float = 0.0
    query_type: str = ""
    chunks_retrieved: int = 0
    retrieval_attempts: int = 0


async def run_agent_with_trace(question: str, user_groups: list[str], vector_store: VectorStore, schema_registry: SchemaRegistry, metadata_store: MetadataStore | None = None) -> tuple[RAGResponse, AgentTrace]:
    import time
    trace = AgentTrace()
    total_start = time.time()

    graph = create_agent_graph(vector_store=vector_store, schema_registry=schema_registry, metadata_store=metadata_store)
    initial_state = AgentState(
        question=question, original_question=question, user_groups=user_groups,
        query_type=None, sub_tasks=[], retrieved_chunks=[], sql_results=[],
        retrieval_attempts=0, needs_reretrieval=False, reformulated_query="",
        answer="", citations=[], warnings=[],
    )

    # Stream through nodes to capture step timings
    step_start = time.time()
    prev_node = None
    async for event in graph.astream(initial_state, stream_mode="updates"):
        now = time.time()
        for node_name, node_output in event.items():
            if prev_node:
                trace.steps.append({"step": prev_node, "time": round(now - step_start, 2), "status": "done"})
            prev_node = node_name
            step_start = now

            # Capture metadata from node outputs
            if node_name == "classify":
                trace.query_type = str(node_output.get("query_type", ""))
            elif node_name == "retrieve":
                trace.chunks_retrieved = len(node_output.get("retrieved_chunks", []))
            elif node_name == "evaluate":
                if node_output.get("needs_reretrieval"):
                    trace.steps.append({"step": "evaluate", "time": round(time.time() - step_start, 2), "status": "re-retrieving"})

    # Final step
    if prev_node:
        trace.steps.append({"step": prev_node, "time": round(time.time() - step_start, 2), "status": "done"})

    trace.total_time = round(time.time() - total_start, 2)
    trace.retrieval_attempts = initial_state.get("retrieval_attempts", 1)

    # Get the final state from the last event
    result = await graph.ainvoke(initial_state)
    trace.chunks_retrieved = len(result.get("retrieved_chunks", []))
    trace.retrieval_attempts = result.get("retrieval_attempts", 1)

    response = RAGResponse(
        answer=result.get("answer", "I could not find any relevant information."),
        citations=result.get("citations", []),
    )
    return response, trace
