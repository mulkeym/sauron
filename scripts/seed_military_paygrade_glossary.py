"""Seed the verified military paygrade -> class glossary for the
payroll_compensation category. Run once against a running instance:

    python scripts/seed_military_paygrade_glossary.py --column col_0

The --column must match the AD pay table's actual key-column name. The spike
found the grade column header is BLANK in the source PDF, so it is named
``col_0`` by _safe_column_names — but ALWAYS confirm against the registered
schema after ingesting the PDF (inspect the schema's first column name). Prefix
patterns cover all grades, so no per-grade enumeration is needed."""
import argparse
import asyncio

PAYGRADE_GLOSSARY = {
    "E-*": "Enlisted Member",
    "O-*": "Commissioned Officer",
    "O-*E": "Commissioned Officer with prior enlisted service",
    "W-*": "Warrant Officer",
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--column", required=True,
                    help="key column name, e.g. 'col_0' (blank header) or 'grade'")
    ap.add_argument("--category", default="payroll_compensation")
    args = ap.parse_args()

    from src.api.routes_ingest import get_metadata_store, get_hint_store
    from src.db.hint_store import SchemaHint
    ms, hs = get_metadata_store(), get_hint_store()
    hint = SchemaHint(
        scope_type="category", scope_value=args.category,
        hint_type="value_glossary", target_column=args.column,
        payload=PAYGRADE_GLOSSARY, provenance="curated", created_by="seed-script")
    await ms.save_hint(hint)
    hs.register(hint)
    print(f"seeded paygrade glossary for column '{args.column}' / category '{args.category}'")


if __name__ == "__main__":
    asyncio.run(main())
