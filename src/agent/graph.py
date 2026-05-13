from __future__ import annotations
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState, QueryType
from src.agent.classifier import classify_query
from src.agent.evaluator import evaluate_context
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
            result = retrieve_sweep(retry_state, vector_store=vector_store)
        elif query_type == QueryType.ANALYTICAL:
            result = await retrieve_analytical(retry_state, vector_store=vector_store, schema_registry=schema_registry)
        elif query_type == QueryType.CROSS_REFERENCE:
            result = await retrieve_cross_reference(retry_state, vector_store=vector_store, schema_registry=schema_registry)
        elif query_type == QueryType.TEMPORAL:
            result = retrieve_lookup(retry_state, vector_store=vector_store)
        else:
            result = retrieve_lookup(retry_state, vector_store=vector_store)

        # Sub-task decomposition: run additional searches for each sub-task
        sub_tasks = state.get("sub_tasks", [])
        if sub_tasks and len(sub_tasks) > 1:
            from src.ingestion.embedder import embed_query
            existing_keys = {(c.metadata.doc_id, c.metadata.chunk_index)
                            for c in result.get("retrieved_chunks", [])}
            added = 0
            for task in sub_tasks:
                if task == state["question"]:
                    continue  # skip if same as main question
                task_vector = embed_query(task)
                task_chunks = vector_store.hybrid_search(
                    vector=task_vector, text_query=task,
                    user_groups=state.get("user_groups", ["ALL"]),
                    top_k=10, tier="small",
                )
                for c in task_chunks:
                    key = (c.metadata.doc_id, c.metadata.chunk_index)
                    if key not in existing_keys:
                        result.setdefault("retrieved_chunks", []).append(c)
                        existing_keys.add(key)
                        added += 1
            if added:
                retrieve_logger.info(f"Sub-task decomposition added {added} chunks from {len(sub_tasks)} sub-tasks")

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

    # Enrich retrieved context with knowledge graph relationships
    async def enrich_with_graph(state: AgentState) -> dict:
        if not metadata_store:
            return {}
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return {}

        # Skip entirely if the knowledge graph is empty — no point making LLM calls
        entity_count = len(await metadata_store.list_entities(limit=1))
        if entity_count == 0:
            return {}

        question = state.get("question", "")
        try:
            import asyncio
            import logging
            from src.generation.llm_client import generate as llm_generate, parse_json_response as parse_json
            from src.ingestion.embedder import embed_query

            kg_logger = logging.getLogger("knowledge_graph")

            # Step 1: Extract key entities from the question via LLM
            extract_resp = await asyncio.to_thread(
                llm_generate,
                system_prompt='Extract the key entity names from this question. Respond with ONLY JSON: {"entities": ["name1", "name2"]}',
                user_prompt=question,
                temperature=0.0, max_tokens=1024,
            )
            search_terms = []
            try:
                parsed = parse_json(extract_resp)
                search_terms = parsed.get("entities", [])
                if isinstance(search_terms, str):
                    search_terms = [search_terms]
            except Exception:
                pass

            if not search_terms:
                return {}

            # Step 2: Query the knowledge graph — find connected entities and their doc/chunk locations
            graph_result = await metadata_store.graph_query(
                entity_names=search_terms,
                depth=2,
            )

            kg_entities = graph_result["entities"]
            kg_rels = graph_result["relationships"]
            kg_doc_chunks = graph_result["doc_chunks"]

            kg_logger.info(f"Graph query for {search_terms}: {len(kg_entities)} entities, {len(kg_rels)} relationships, {len(kg_doc_chunks)} doc/chunk pairs")

            if not kg_entities:
                return {}

            # Step 3: Build a knowledge graph summary as a synthetic chunk
            kg_lines = ["Knowledge Graph Context:"]
            seen_rels = set()
            for r in kg_rels:
                rel_key = (r["source"], r["relationship"], r["target"])
                if rel_key in seen_rels:
                    continue
                seen_rels.add(rel_key)
                kg_lines.append(f"  {r['source']} ({r['source_type']}) —[{r['relationship']}]→ {r['target']} ({r['target_type']})")
            if len(kg_lines) > 1:
                kg_chunk = RetrievedChunk(
                    text="\n".join(kg_lines), score=0.5,
                    metadata=ChunkMetadata(
                        doc_id="knowledge-graph", filename="knowledge_graph",
                        doc_type="graph", chunk_index=0, start_char=0, acl_groups=["ALL"],
                    ),
                )
                chunks = chunks + [kg_chunk]

            # Step 4: Pull additional document chunks where graph entities were mentioned
            # This is the key improvement — use entity mention locations to find chunks
            # that the initial vector search might have missed
            existing_keys = {(c.metadata.doc_id, c.metadata.chunk_index) for c in chunks}
            new_chunk_locations = [(doc_id, chunk_idx) for doc_id, chunk_idx in kg_doc_chunks
                                  if (doc_id, chunk_idx) not in existing_keys]

            if new_chunk_locations:
                # Retrieve these specific chunks from the vector store by searching
                # with the entity names as keywords
                entity_names_str = " ".join(e["name"] for e in kg_entities[:5])
                query_vector = await asyncio.to_thread(embed_query, entity_names_str)
                extra_chunks = vector_store.search(
                    vector=query_vector, user_groups=state.get("user_groups", ["ALL"]),
                    top_k=min(len(new_chunk_locations), 20),
                )
                # Only add chunks that are from the graph-discovered doc/chunk pairs
                graph_doc_ids = {doc_id for doc_id, _ in new_chunk_locations}
                added = 0
                for ec in extra_chunks:
                    key = (ec.metadata.doc_id, ec.metadata.chunk_index)
                    if ec.metadata.doc_id in graph_doc_ids and key not in existing_keys:
                        chunks.append(ec)
                        existing_keys.add(key)
                        added += 1
                if added:
                    kg_logger.info(f"Added {added} additional chunks from graph entity mentions")

            return {"retrieved_chunks": chunks}
        except Exception as e:
            import logging
            logging.getLogger("knowledge_graph").warning(f"Graph enrichment failed: {e}")
            return {}

    graph.add_node("enrich", enrich_with_graph)
    graph.add_node("evaluate", evaluate_context)
    graph.add_node("synthesize", synthesize_answer)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "enrich")
    graph.add_edge("enrich", "evaluate")

    def should_reretrieval(state: AgentState) -> str:
        if state.get("needs_reretrieval", False):
            return "retrieve"
        return "synthesize"

    graph.add_conditional_edges("evaluate", should_reretrieval, {"retrieve": "retrieve", "synthesize": "synthesize"})
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
