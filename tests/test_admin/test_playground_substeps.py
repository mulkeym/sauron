"""Tests for the playground's live classify sub-step plumbing."""
import src.admin.routes as routes
from src.admin.routes import (
    _classify_substep_label, _record_substep, _format_classify_detail,
)


def test_classify_substep_label_maps_sub_steps():
    assert _classify_substep_label("classify.hints") == "reading available data tables"
    assert _classify_substep_label("classify.llm") == "classifying question"
    assert _classify_substep_label("classify.strategy") == "checking strategy memory"


def test_classify_substep_label_empty_for_done_and_others():
    assert _classify_substep_label("classify.done") == ""
    assert _classify_substep_label("retrieve") == ""
    assert _classify_substep_label("classify") == ""


def test_record_substep_sets_and_clears_active_substep():
    routes._playground_jobs["q"] = {"active_substep": ""}
    _record_substep("q", "classify.strategy")
    assert routes._playground_jobs["q"]["active_substep"] == "checking strategy memory"
    _record_substep("q", "classify.done")              # done clears it
    assert routes._playground_jobs["q"]["active_substep"] == ""
    del routes._playground_jobs["q"]


def test_record_substep_unknown_query_is_noop():
    _record_substep("does-not-exist", "classify.llm")  # must not raise


def test_format_classify_detail_shows_type_reason_subtasks_and_override():
    html = _format_classify_detail({
        "query_type": "analytical", "reason": "asks for pay by grade",
        "sub_tasks": ["gs-13 pay"],
        "strategy_memory": {"overrode": True, "llm_pick": "lookup",
                            "memory_best": "analytical", "count": 7, "margin": 0.8},
    })
    assert "analytical" in html
    assert "asks for pay by grade" in html
    assert "gs-13 pay" in html
    assert "override" in html.lower()


def test_format_classify_detail_kept_decision():
    html = _format_classify_detail({
        "query_type": "lookup", "reason": "a fact lookup", "sub_tasks": [],
        "strategy_memory": {"overrode": False, "llm_pick": "lookup", "reason": "no record"},
    })
    assert "lookup" in html
    assert "kept" in html.lower()
