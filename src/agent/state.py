from __future__ import annotations
from enum import StrEnum
from typing import Annotated, TypedDict
from src.retrieval.models import Citation, RetrievedChunk


def chunk_key(chunk: RetrievedChunk) -> tuple:
    m = chunk.metadata
    return (m.doc_id, m.chunk_size_tier, m.chunk_index, m.start_char,
            m.content_type, m.figure_id)


def _merge_chunks(existing: list[RetrievedChunk], new: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep distinct source spans and size tiers when combining branches."""
    seen = {chunk_key(c) for c in existing}
    merged = list(existing)
    for c in new:
        key = chunk_key(c)
        if key not in seen:
            merged.append(c)
            seen.add(key)
    return merged


class QueryType(StrEnum):
    PROCEDURE = "procedure"
    TROUBLESHOOTING = "troubleshooting"
    LOOKUP = "lookup"
    SWEEP = "sweep"
    ANALYTICAL = "analytical"
    CROSS_REFERENCE = "cross_reference"
    TEMPORAL = "temporal"
    METADATA = "metadata"

class AgentState(TypedDict, total=False):
    question: str
    diagram_discovery: bool  # direct search/list/show of stored figures
    original_question: str  # preserved across retries
    user_groups: list[str]
    query_type: QueryType | None
    reason: str
    sub_tasks: list[str]
    retrieved_chunks: Annotated[list[RetrievedChunk], _merge_chunks]
    sql_results: list[dict]
    retrieval_attempts: int
    needs_reretrieval: bool
    reformulated_query: str  # alternative query for retry
    answer: str
    citations: list[Citation]
    warnings: list[str]
    answer_profile: dict  # immutable request snapshot of one published/draft profile
    response_kind: str  # answer, clarification, or insufficient_evidence
    conversation: list[dict]
    edition_decisions: dict
    revision_missing_details: list[str]
    technical_intent: str
    graph_retrieval_hints: list[dict]
    graph_retrieval_trace: dict
    technical_coverage: dict
    technical_context_keys: list[str]  # bounded original neighbors retained through final reranking
    preview: bool
    preview_evidence: list[dict]
    skip_graph: bool
    allowed_doc_ids: list[str]  # restrict retrieval to these doc_ids (app filter)
    dataset_id: int  # dataset filter for KG queries
    structured_trace: dict  # playground: structured/SQL lookup decision + result
    feedback_boosts: dict[str, float]  # {doc_id: boost} from relevance feedback, passed to final rerank
    strategy_memory: dict  # routing decision from Strategy Memory (F1 observability)
    progress: object  # optional Callable[[str, dict|None], None]: live sub-step reporter for async-status visibility; declared so LangGraph keeps it as a state channel (no-op when absent)
