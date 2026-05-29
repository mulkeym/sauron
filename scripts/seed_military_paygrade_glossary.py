"""Seed the verified military paygrade glossary + domain notes for the AD pay
table. Run against a running instance, scoping to the doc's STABLE dataset_id
(auto-category is non-deterministic, so prefer --dataset):

    python scripts/seed_military_paygrade_glossary.py --dataset 2 --column col_0

The --column must match the AD pay table's actual key-column name. The grade
column header is BLANK in the source PDF, so _safe_column_names names it
``col_0`` — confirm against the registered schema (its first column) post-ingest.

IMPORTANT: seed BEFORE re-ingesting the PDF — row narratives are annotated at
ingest time. The text-to-SQL prompt reads hints live, so that half works anytime.
"""
import argparse
import asyncio

# glossary_lookup matches EXACT keys first, then KEY*-prefix patterns. The
# "prior enlisted service" grades (O-1E/O-2E/O-3E) MUST be exact entries so they
# win over the "O-*" prefix — a "O-*E" pattern would never fire (patterns must
# end in "*"), leaving those grades mislabeled "Commissioned Officer".
PAYGRADE_GLOSSARY = {
    "O-1E": "Commissioned Officer with prior enlisted service",
    "O-2E": "Commissioned Officer with prior enlisted service",
    "O-3E": "Commissioned Officer with prior enlisted service",
    "E-*": "Enlisted Member",
    "O-*": "Commissioned Officer",
    "W-*": "Warrant Officer",
}

TABLE_NOTE = ("U.S. military active-duty monthly basic pay: rows are pay grades "
              "(Commissioned Officer O-*, Warrant Officer W-*, Enlisted Member E-*) "
              "and columns are years-of-service brackets.")
COLUMN_NOTE = "Military pay grade (O-* officer, W-* warrant, E-* enlisted)."


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--column", required=True,
                    help="key column name, e.g. 'col_0' (blank header) or 'grade'")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--dataset", type=int, help="dataset_id scope (preferred — stable)")
    grp.add_argument("--category", help="category scope (fragile — auto-categorization varies)")
    args = ap.parse_args()

    if args.dataset is not None:
        scope_type, scope_value = "dataset", str(args.dataset)
    elif args.category:
        scope_type, scope_value = "category", args.category
    else:
        ap.error("provide --dataset (preferred) or --category")

    from src.api.routes_ingest import get_metadata_store, get_hint_store
    from src.db.hint_store import SchemaHint
    ms, hs = get_metadata_store(), get_hint_store()

    hints = [
        SchemaHint(scope_type=scope_type, scope_value=scope_value,
                   hint_type="value_glossary", target_column=args.column,
                   payload=PAYGRADE_GLOSSARY, provenance="curated", created_by="seed-script"),
        SchemaHint(scope_type=scope_type, scope_value=scope_value,
                   hint_type="column_note", target_column=args.column,
                   payload={"text": COLUMN_NOTE}, provenance="curated", created_by="seed-script"),
        SchemaHint(scope_type=scope_type, scope_value=scope_value,
                   hint_type="table_note", target_column=None,
                   payload={"text": TABLE_NOTE}, provenance="curated", created_by="seed-script"),
    ]
    for h in hints:
        await ms.save_hint(h)
        hs.register(h)
    print(f"seeded {len(hints)} paygrade hints (value_glossary + column_note + table_note) "
          f"for column '{args.column}' / {scope_type}={scope_value}")


if __name__ == "__main__":
    asyncio.run(main())
