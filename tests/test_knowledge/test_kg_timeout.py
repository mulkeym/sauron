"""KG extract timeout estimation."""
from src.config import settings
from src.knowledge.graph_rag import estimate_kg_timeout_seconds


def test_adaptive_timeout_scales_with_text(monkeypatch):
    monkeypatch.setattr(settings, "kg_extract_timeout_seconds", 0)
    monkeypatch.setattr(settings, "kg_extract_timeout_min_seconds", 900)
    monkeypatch.setattr(settings, "kg_extract_timeout_max_seconds", 3600)
    monkeypatch.setattr(settings, "kg_chunk_token_size", 1200)
    monkeypatch.setattr(settings, "llm_concurrency", 2)
    monkeypatch.setattr(settings, "kg_extract_sec_per_chunk", 22.0)

    small = estimate_kg_timeout_seconds("x" * 1000)
    large = estimate_kg_timeout_seconds("x" * 400_000)
    assert small >= 900
    assert large > small
    assert large <= 3600


def test_fixed_timeout_override(monkeypatch):
    monkeypatch.setattr(settings, "kg_extract_timeout_seconds", 1234)
    assert estimate_kg_timeout_seconds("anything") == 1234
