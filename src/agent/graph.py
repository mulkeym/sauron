from __future__ import annotations
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState, QueryType
from src.agent.classifier import _classify_node_factory
from src.agent.synthesizer import synthesize_answer
from src.agent.strategies.lookup import retrieve_lookup
from src.agent.strategies.sweep import retrieve_sweep
from src.agent.strategies.map_reduce import retrieve_map_reduce
from src.agent.strategies.analytical import retrieve_analytical
from src.agent.strategies.structured import retrieve_structured
from src.agent.strategies.cross_reference import retrieve_cross_reference
from src.db.metadata import MetadataStore
from src.db.schema_registry import SchemaRegistry
from src.generation.rag_chain import RAGResponse
from src.retrieval.models import RetrievedChunk, ChunkMetadata
from src.retrieval.vector_store import VectorStore


def _skip_enrich(state) -> bool:
    """True when knowledge-graph enrichment should be skipped: explicitly via
    skip_graph, or for METADATA (catalog) queries where graph context is noise."""
    from src.agent.state import QueryType
    if state.get("skip_graph"):
        return True
    return state.get("query_type") == QueryType.METADATA


def _rerank_merge(state, vector_store) -> dict:
    """Final-N rerank of the consolidated chunk set (mutates chunk.score in
    place; consumers sort by score). Fail-open + flag-guarded."""
    import logging
    from src.config import settings
    if not settings.rerank_final_enabled:
        return {}
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {}
    boosts = state.get("feedback_boosts", {}) or {}
    try:
        vector_store.rerank_chunks(
            chunks, state.get("question", ""), settings.rerank_final_top_n, boosts=boosts,
        )
    except Exception as e:
        logging.getLogger("retrieval").warning(f"Final rerank skipped: {e}")
    return {}


def create_agent_graph(vector_store: VectorStore, schema_registry: SchemaRegistry, metadata_store: MetadataStore | None = None, include_synthesize: bool = True):
    graph = StateGraph(AgentState)

    graph.add_node("classify", _classify_node_factory(schema_registry))

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
            import asyncio as _asyncio_lookup
            result = await _asyncio_lookup.to_thread(retrieve_lookup, retry_state, vector_store=vector_store)
        elif query_type == QueryType.SWEEP:
            # Run both sweep (raw chunks) and map-reduce (per-doc extraction), merge results
            import asyncio as _asyncio
            sweep_result, mr_result, struct_result = await _asyncio.gather(
                retrieve_sweep(retry_state, vector_store=vector_store),
                retrieve_map_reduce(retry_state, vector_store=vector_store),
                retrieve_structured(retry_state, vector_store=vector_store, schema_registry=schema_registry),
            )
            # Merge: map-reduce synthetic chunk + sweep raw chunks (deduplicated)
            merged_chunks = mr_result.get("retrieved_chunks", [])
            seen_keys = {(c.metadata.doc_id, c.metadata.chunk_index) for c in merged_chunks}
            for c in sweep_result.get("retrieved_chunks", []):
                key = (c.metadata.doc_id, c.metadata.chunk_index)
                if key not in seen_keys:
                    merged_chunks.append(c)
                    seen_keys.add(key)
            for c in struct_result.get("retrieved_chunks", []):
                key = (c.metadata.doc_id, c.metadata.chunk_index)
                if key not in seen_keys:
                    merged_chunks.append(c)
                    seen_keys.add(key)
            # Stamp map-reduce's normalized relevance onto the raw chunks so the
            # playground and citations show a meaningful score instead of the
            # 0.0 that fetched-by-doc-id chunks carry.
            doc_relevance = mr_result.get("doc_relevance", {})
            if doc_relevance:
                SYNTHETIC_IDS = {"map-reduce", "knowledge-graph", "metadata-context"}
                for c in merged_chunks:
                    did = c.metadata.doc_id
                    if did not in SYNTHETIC_IDS and did in doc_relevance:
                        c.score = doc_relevance[did]
            retrieve_logger.info(f"Sweep+MapReduce merged: {len(merged_chunks)} total chunks")
            result = {"retrieved_chunks": merged_chunks, "retrieval_attempts": retry_state.get("retrieval_attempts", 0) + 1}
            if struct_result.get("sql_results"):
                result["sql_results"] = struct_result["sql_results"]
            if struct_result.get("structured_trace"):
                result["structured_trace"] = struct_result["structured_trace"]
            if mr_result.get("feedback_boosts"):
                result["feedback_boosts"] = mr_result["feedback_boosts"]
            # Add lightweight metadata context from documents not fully MAP'd
            try:
                from src.api.routes_ingest import get_metadata_store
                _ms = get_metadata_store()
                sweep_doc_ids = {c.metadata.doc_id for c in sweep_result.get("retrieved_chunks", [])}
                mr_doc_ids = set()
                for c in mr_result.get("retrieved_chunks", []):
                    if c.metadata.doc_id != "map-reduce":
                        mr_doc_ids.add(c.metadata.doc_id)
                extra_doc_ids = sweep_doc_ids - mr_doc_ids - {"map-reduce", "knowledge-graph"}
                if extra_doc_ids:
                    meta_parts = []
                    for did in list(extra_doc_ids)[:20]:
                        doc_rec = await _ms.get_document(did)
                        if doc_rec and getattr(doc_rec, 'metadata_tags', None):
                            meta = doc_rec.metadata_tags
                            parts = []
                            for field in ["entities", "organizations", "amounts", "identifiers", "topics"]:
                                vals = meta.get(field, [])
                                if vals:
                                    parts.append(f"{field}: {', '.join(vals[:5])}")
                            if parts:
                                meta_parts.append(f"[{doc_rec.filename}]: {'; '.join(parts)}")
                    if meta_parts:
                        meta_chunk = RetrievedChunk(
                            text=f"Additional document metadata ({len(meta_parts)} docs not fully analyzed):\n" + "\n".join(meta_parts),
                            score=0.3,
                            metadata=ChunkMetadata(
                                doc_id="metadata-context", filename="metadata_context",
                                doc_type="metadata", chunk_index=0, start_char=0, acl_groups=["ALL"],
                            ),
                        )
                        merged_chunks.append(meta_chunk)
                        retrieve_logger.info(f"Added metadata context from {len(meta_parts)} additional documents")
            except Exception as e:
                retrieve_logger.debug(f"Metadata context enrichment skipped: {e}")
        elif query_type == QueryType.ANALYTICAL:
            result = await retrieve_analytical(retry_state, vector_store=vector_store, schema_registry=schema_registry)
        elif query_type == QueryType.METADATA:
            from src.agent.strategies.metadata_catalog import retrieve_metadata_catalog
            result = await retrieve_metadata_catalog(retry_state, metadata_store=metadata_store)
        elif query_type == QueryType.CROSS_REFERENCE:
            result = await retrieve_cross_reference(retry_state, vector_store=vector_store, schema_registry=schema_registry)
        elif query_type == QueryType.TEMPORAL:
            result = await _asyncio_lookup.to_thread(retrieve_lookup, retry_state, vector_store=vector_store)
        else:
            result = await _asyncio_lookup.to_thread(retrieve_lookup, retry_state, vector_store=vector_store)

        # Sub-task decomposition: run additional searches for each sub-task IN PARALLEL.
        # Skipped for METADATA: catalog answers come entirely from the catalog SQL, so
        # content vector chunks would only add noise/citations.
        sub_tasks = state.get("sub_tasks", [])
        unique_tasks = [t for t in sub_tasks if t != state["question"]] if sub_tasks else []
        if unique_tasks and query_type != QueryType.METADATA:
            from src.ingestion.embedder import embed_texts
            import asyncio

            # Embed all sub-tasks in one batch (thread-safe, no concurrent model access)
            task_vectors = await asyncio.to_thread(embed_texts, unique_tasks, "query")

            # Search in parallel (safe — LanceDB handles concurrent reads)
            sub_doc_ids = state.get("allowed_doc_ids")
            async def search_subtask(task, vector):
                return await asyncio.to_thread(
                    vector_store.hybrid_search,
                    vector=vector, text_query=task,
                    user_groups=state.get("user_groups", ["ALL"]),
                    top_k=10, tier="small",
                    doc_ids=sub_doc_ids,
                )

            all_subtask_results = await asyncio.gather(
                *[search_subtask(t, v) for t, v in zip(unique_tasks, task_vectors)]
            )

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

    # Knowledge graph query — runs in parallel with retrieve (only needs question + user_groups)
    async def enrich_with_graph(state: AgentState) -> dict:
        if _skip_enrich(state):
            return {}

        question = state.get("question", "")
        try:
            import logging
            from src.knowledge.graph_rag import query_graph, is_graph_populated

            kg_logger = logging.getLogger("knowledge_graph")

            if not await is_graph_populated():
                return {}

            user_groups = state.get("user_groups", ["ALL"])
            ds_id = state.get("dataset_id", 0)
            result = await query_graph(question, mode="mix", user_groups=user_groups, dataset_id=ds_id)
            kg_context = result.get("context", "")

            if not kg_context or len(kg_context.strip()) < 20:
                return {}

            kg_logger.info(f"LightRAG enrichment: {len(kg_context)} chars of graph context")

            kg_chunk = RetrievedChunk(
                text=f"Knowledge Graph Context:\n{kg_context}",
                score=0.5,
                metadata=ChunkMetadata(
                    doc_id="knowledge-graph", filename="knowledge_graph",
                    doc_type="graph", chunk_index=0, start_char=0, acl_groups=["ALL"],
                ),
            )
            return {"retrieved_chunks": [kg_chunk]}
        except Exception as e:
            import logging
            logging.getLogger("knowledge_graph").warning(f"Graph enrichment failed: {e}")
            return {}

    graph.add_node("enrich", enrich_with_graph)

    # Merge node: combine retrieve + enrich results before synthesis
    def merge_results(state: AgentState) -> dict:
        """Final-N rerank over the chunks both branches produced (mutates
        scores in place; the additive reducer means we return {})."""
        return _rerank_merge(state, vector_store)

    graph.add_node("merge", merge_results)

    graph.set_entry_point("classify")
    # After classify, run retrieve and enrich IN PARALLEL
    graph.add_edge("classify", "retrieve")
    graph.add_edge("classify", "enrich")
    # Both feed into merge, then synthesize
    graph.add_edge("retrieve", "merge")
    graph.add_edge("enrich", "merge")
    # The playground streams the answer separately, so it builds the graph with
    # include_synthesize=False and finishes at merge (no answer produced in-graph).
    if include_synthesize:
        graph.add_node("synthesize", synthesize_answer)
        graph.add_edge("merge", "synthesize")
        graph.add_edge("synthesize", END)
    else:
        graph.add_edge("merge", END)
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
    strategy_memory: dict | None = None


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
                trace.strategy_memory = node_output.get("strategy_memory")
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
