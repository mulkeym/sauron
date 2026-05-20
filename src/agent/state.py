from __future__ import annotations
from enum import StrEnum
from typing import Annotated, TypedDict
from src.retrieval.models import Citation, RetrievedChunk


def _merge_chunks(existing: list[RetrievedChunk], new: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reducer: merge retrieved_chunks from parallel branches, deduplicating by (doc_id, chunk_index)."""
    seen = {(c.metadata.doc_id, c.metadata.chunk_index) for c in existing}
    merged = list(existing)
    for c in new:
        key = (c.metadata.doc_id, c.metadata.chunk_index)
        if key not in seen:
            merged.append(c)
            seen.add(key)
    return merged


class QueryType(StrEnum):
    LOOKUP = "lookup"
    SWEEP = "sweep"
    ANALYTICAL = "analytical"
    CROSS_REFERENCE = "cross_reference"
    TEMPORAL = "temporal"

class AgentState(TypedDict, total=False):
    question: str
    original_question: str  # preserved across retries
    user_groups: list[str]
    query_type: QueryType | None
    sub_tasks: list[str]
    retrieved_chunks: Annotated[list[RetrievedChunk], _merge_chunks]
    sql_results: list[dict]
    retrieval_attempts: int
    needs_reretrieval: bool
    reformulated_query: str  # alternative query for retry
    answer: str
    citations: list[Citation]
    warnings: list[str]
    skip_graph: bool
    allowed_doc_ids: list[str]  # restrict retrieval to these doc_ids (app filter)
