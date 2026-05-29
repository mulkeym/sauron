import pytest
from pathlib import Path
from src.ingestion.pdf_extract import extract_pdf
from src.ingestion import tabular_ingest as ti
from src.db.hint_store import HintStore, SchemaHint

FIX = Path("tests/fixtures/pdf/two_page_table.pdf")


class _VS:
    def __init__(self): self.upserts = []
    def upsert(self, texts, vectors, metadatas):
        self.upserts.append((list(texts), list(metadatas)))


class _MS:
    async def save_schema(self, schema): pass


class _Reg:
    def register(self, schema): pass


@pytest.mark.skipif(not FIX.exists(), reason="fixture missing")
@pytest.mark.asyncio
async def test_pdf_to_structured_with_glossary(tmp_path, monkeypatch):
    """PDF -> extract_pdf -> ingest_grids yields glossary-annotated row narratives
    that mention 'Enlisted'/'Officer' (findable by the original failing query)."""
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))

    monkeypatch.setattr("src.ingestion.tabular_ingest.embed_texts",
                        lambda texts, *a, **k: [[0.0, 0.0, 0.0] for _ in texts])
    hs = HintStore()
    hs.register(SchemaHint(
        scope_type="category", scope_value="payroll_compensation",
        hint_type="value_glossary", target_column="grade",
        payload={"E-*": "Enlisted Member", "O-*": "Commissioned Officer"}))
    monkeypatch.setattr("src.api.routes_ingest.get_hint_store", lambda: hs)

    extracted = extract_pdf(FIX)
    assert extracted.table_grids, "expected at least one table grid"

    vs, ms, reg = _VS(), _MS(), _Reg()
    classifications, ingested = await ti.ingest_grids(
        extracted.table_grids, "docE2E", "ad.pdf", "pdf",
        ["executives"], "payroll_compensation",
        vs, ms, schema_registry=reg,
        generate_fn=lambda **k: '{"key_columns":["grade"],"measure_columns":["over2","over4"],'
                                '"column_descriptions":{},"table_description":"AD pay"}',
    )
    joined = "\n".join(t for texts, _ in vs.upserts for t in texts)
    assert "Enlisted Member" in joined        # E-1/E-3 rows annotated
    assert "Commissioned Officer" in joined    # O-1/O-2 rows annotated
