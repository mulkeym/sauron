from __future__ import annotations
from enum import StrEnum
from typing import TypedDict
from src.retrieval.models import Citation, RetrievedChunk

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
    retrieved_chunks: list[RetrievedChunk]
    sql_results: list[dict]
    retrieval_attempts: int
    needs_reretrieval: bool
    reformulated_query: str  # alternative query for retry
    answer: str
    citations: list[Citation]
    warnings: list[str]
