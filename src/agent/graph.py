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
        query_type = state.get("query_type", QueryType.LOOKUP)
        attempts = state.get("retrieval_attempts", 0)

        # On retry, use reformulated query if available
        if attempts > 0 and state.get("reformulated_query"):
            # Create a modified state with the new question for retrieval
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
        question = state.get("question", "")
        try:
            # Use LLM to extract the key entity from the question
            import asyncio
            from src.generation.llm_client import generate as llm_generate, parse_json_response as parse_json
            extract_resp = await asyncio.to_thread(
                llm_generate,
                system_prompt='Extract the main entity name from this question. Respond with ONLY JSON: {"entity": "name"}',
                user_prompt=question,
                temperature=0.0, max_tokens=128,
            )
            try:
                parsed = parse_json(extract_resp)
                search_term = parsed.get("entity", "")
            except Exception:
                search_term = ""

            # Search knowledge graph with extracted entity
            entities = []
            if search_term:
                entities = await metadata_store.search_entities(search_term)
            # Fallback: try individual capitalized words
            if not entities:
                for word in question.split():
                    if len(word) > 3 and word[0].isupper():
                        entities = await metadata_store.search_entities(word)
                        if entities:
                            break
            if entities:
                details = await metadata_store.get_entity_details(entities[0].id)
                if details["relationships"]:
                    # Add knowledge graph context as a synthetic chunk
                    kg_text = f"Knowledge Graph for '{entities[0].name}' ({entities[0].entity_type}):\n"
                    for r in details["relationships"]:
                        kg_text += f"  - {r['relationship_type']} → {r['related_entity']} ({r['entity_type']})\n"
                    for m in details["mentions"][:3]:
                        if m["context_snippet"]:
                            kg_text += f"  Mentioned in: {m['context_snippet'][:150]}\n"
                    kg_chunk = RetrievedChunk(
                        text=kg_text, score=0.5,
                        metadata=ChunkMetadata(
                            doc_id="knowledge-graph", filename="knowledge_graph",
                            doc_type="graph", chunk_index=0, start_char=0, acl_groups=["ALL"],
                        ),
                    )
                    return {"retrieved_chunks": chunks + [kg_chunk]}
        except Exception:
            pass
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
