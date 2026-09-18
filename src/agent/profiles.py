"""Validated answer policy, captured once for each request."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnswerProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    instructions: str = Field(default="", max_length=20000)
    routing_instructions: str = Field(default="", max_length=6000)
    strategy: Literal["auto", "lookup", "sweep", "analytical", "cross_reference", "temporal", "metadata"] = "auto"
    retrieval_depth: Literal["focused", "balanced", "thorough"] = "balanced"
    graph_enrichment: bool = True
    structured_lookup: bool = True
    strategy_memory: bool = False
    max_subtasks: int = Field(default=3, ge=0, le=8)
    clarification: Literal["when_needed", "answer_with_caveats"] = "when_needed"
    clarification_fields: list[Literal["platform", "software_version", "environment", "site"]] = Field(
        default_factory=lambda: ["platform", "software_version", "environment", "site"], max_length=4)
    insufficient_evidence: Literal["partial", "abstain"] = "partial"

    @model_validator(mode="after")
    def consistent_policy(self):
        if self.strategy == "analytical" and not self.structured_lookup:
            raise ValueError("Analytical routing requires structured lookup.")
        if len(self.clarification_fields) != len(set(self.clarification_fields)):
            raise ValueError("Clarification fields must be unique.")
        return self


DEPTH_LIMITS = {
    "focused": {"lookup": 15, "discovery": 50, "subtask": 5, "window": 1},
    "balanced": {"lookup": 30, "discovery": 250, "subtask": 10, "window": 2},
    "thorough": {"lookup": 60, "discovery": 500, "subtask": 20, "window": 3},
}
CLARIFICATION_QUESTIONS = {
    "platform": "Which SD-WAN platform or product does this apply to?",
    "software_version": "Which software or firmware version are you using?",
    "environment": "Which environment does this apply to, such as lab or production?",
    "site": "Which site or deployment does this apply to?",
}


def profile_for_state(state) -> AnswerProfile | None:
    snapshot = state.get("answer_profile")
    return AnswerProfile.model_validate(snapshot["config"]) if snapshot else None


def retrieval_limit(state, name, fallback):
    profile = profile_for_state(state)
    return DEPTH_LIMITS[profile.retrieval_depth][name] if profile else fallback


def structured_enabled(state):
    profile = profile_for_state(state)
    return profile.structured_lookup if profile else True


def snapshot(profile_id, revision, config):
    from src.config import settings
    return {"profile_id": profile_id, "revision": revision,
            "config": AnswerProfile.model_validate(config).model_dump(),
            "team_instructions": settings.answer_domain_instructions}


def active_snapshot():
    from src.agent.profile_store import read_book
    book = read_book()
    active = book["active"]
    revision = next(r for r in book["profiles"][active["profile_id"]]["revisions"]
                    if r["revision"] == active["revision"])
    return snapshot(active["profile_id"], active["revision"], revision["config"])


def response_policy(profile: AnswerProfile) -> str:
    clarification = (
        "When a missing detail materially changes the documented procedure, ask a clarification before giving steps. "
        "Only request missing details from this list: " + ", ".join(profile.clarification_fields) + "."
        if profile.clarification == "when_needed" and profile.clarification_fields
        else "Do not ask a follow-up question. Give only supported information with explicit caveats; never guess missing details."
    )
    evidence = (
        "If the evidence cannot support the complete requested answer, abstain instead of giving a partial procedure."
        if profile.insufficient_evidence == "abstain"
        else "You may give a partial answer using cited evidence. Explicitly identify what the evidence does not establish."
    )
    return clarification + "\n" + evidence + """
Return a JSON object, without a code fence, using one of these forms:
{"status":"answer","answer":"Your answer with exact [E...] evidence citations."}
{"status":"clarification","missing_details":["platform","software_version"]}
{"status":"insufficient_evidence"}
For clarification, return only the missing detail identifiers; the application supplies the questions.
Use insufficient_evidence when you cannot provide a grounded answer under the policy above.
Do not include factual claims in a clarification or insufficient_evidence response."""
