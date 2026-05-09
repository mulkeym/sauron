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
from src.db.schema_registry import SchemaRegistry
from src.generation.rag_chain import RAGResponse
from src.retrieval.vector_store import VectorStore


def create_agent_graph(vector_store: VectorStore, schema_registry: SchemaRegistry):
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_query)

    async def retrieve(state: AgentState) -> dict:
        query_type = state.get("query_type", QueryType.LOOKUP)
        if query_type == QueryType.LOOKUP:
            return retrieve_lookup(state, vector_store=vector_store)
        elif query_type == QueryType.SWEEP:
            return retrieve_sweep(state, vector_store=vector_store)
        elif query_type == QueryType.ANALYTICAL:
            return await retrieve_analytical(state, schema_registry=schema_registry)
        elif query_type == QueryType.CROSS_REFERENCE:
            return await retrieve_cross_reference(state, vector_store=vector_store, schema_registry=schema_registry)
        elif query_type == QueryType.TEMPORAL:
            return retrieve_lookup(state, vector_store=vector_store)
        return retrieve_lookup(state, vector_store=vector_store)

    graph.add_node("retrieve", retrieve)
    graph.add_node("evaluate", evaluate_context)
    graph.add_node("synthesize", synthesize_answer)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "evaluate")

    def should_reretrieval(state: AgentState) -> str:
        if state.get("needs_reretrieval", False):
            return "retrieve"
        return "synthesize"

    graph.add_conditional_edges("evaluate", should_reretrieval, {"retrieve": "retrieve", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)
    return graph.compile()


async def run_agent(question: str, user_groups: list[str], vector_store: VectorStore, schema_registry: SchemaRegistry) -> RAGResponse:
    graph = create_agent_graph(vector_store=vector_store, schema_registry=schema_registry)
    initial_state = AgentState(
        question=question, user_groups=user_groups, query_type=None, sub_tasks=[],
        retrieved_chunks=[], sql_results=[], retrieval_attempts=0,
        needs_reretrieval=False, answer="", citations=[], warnings=[],
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


async def run_agent_with_trace(question: str, user_groups: list[str], vector_store: VectorStore, schema_registry: SchemaRegistry) -> tuple[RAGResponse, AgentTrace]:
    import time
    trace = AgentTrace()
    total_start = time.time()

    graph = create_agent_graph(vector_store=vector_store, schema_registry=schema_registry)
    initial_state = AgentState(
        question=question, user_groups=user_groups, query_type=None, sub_tasks=[],
        retrieved_chunks=[], sql_results=[], retrieval_attempts=0,
        needs_reretrieval=False, answer="", citations=[], warnings=[],
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
