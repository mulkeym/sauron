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
    strategy: Literal["auto", "lookup", "procedure", "troubleshooting", "sweep", "analytical", "cross_reference", "temporal", "metadata"] = "auto"
    retrieval_depth: Literal["focused", "balanced", "thorough"] = "balanced"
    images: Literal["inherit", "auto", "requested", "off"] = "inherit"
    max_images: int = Field(default=2, ge=0, le=5)
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
    clarify = profile.clarification == "when_needed" and bool(profile.clarification_fields)
    decision = (
        "First, request clarification only if a missing detail from " + ", ".join(profile.clarification_fields)
        + " changes the requested procedure. Otherwise, apply the evidence policy below."
        if clarify else "Do not request clarification. Apply the evidence policy below."
    )
    evidence = (
        "Answer only if evidence supports the complete requested answer; otherwise abstain."
        if profile.insufficient_evidence == "abstain"
        else "Answer the supported part and state any gaps. Abstain only when no useful part of the answer is supported."
    )
    lines = [decision, evidence, "Response format — use exactly one of these forms:",
             "SAURON_STATUS: answer\n<Markdown answer with [E...] citations>"]
    if clarify:
        lines.append("SAURON_STATUS: clarification\n<Comma-separated identifiers from: "
                     + ", ".join(profile.clarification_fields) + ">")
    lines += ["SAURON_STATUS: insufficient_evidence",
              "Abstention has no body. Write Markdown directly. No JSON wrapper. Keep internal reasoning out of the visible final response."]
    return "\n\n".join(lines)
