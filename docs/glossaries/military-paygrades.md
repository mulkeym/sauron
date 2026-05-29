# Military paygrade glossary

Maps DoD pay grades to personnel class for the `payroll_compensation` category.
Verified against the standard DoD pay-grade scheme:
- `E-1`..`E-9` → Enlisted Member
- `O-1`..`O-10` → Commissioned Officer
- `O-1E`..`O-3E` → Commissioned Officer with prior enlisted service
- `W-1`..`W-5` → Warrant Officer

Prefix patterns (`E-*`, `O-*`, `W-*`, `O-*E`) are used so all grades resolve
without enumerating each. Seeded via `scripts/seed_military_paygrade_glossary.py`.

The source AD pay PDF's grade column has a **blank header**, so it is named
`col_0` by `_safe_column_names`. Pass `--column col_0` (confirm against the
registered schema after ingesting the PDF — inspect the first column's name).

Lesson from the locality glossary: glossary content is operator-verified, never
model-guessed (the model once wrongly claimed `TU`=Tampa; it is Tucson).
