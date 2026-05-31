# Schema Hints Viewer (read-only admin page)

**Date:** 2026-05-31
**Status:** Design approved, pending spec review

## Problem

SQL "hints" (domain knowledge injected into the text-to-SQL schema prompt — value glossaries like `AK → State of Alaska`, plus column/table notes) are currently invisible: they exist only in the `schema_hints` table and are created solely by seed scripts (e.g. `scripts/seed_military_paygrade_glossary.py`). There's no way to see what hints exist or what they map. As new data types are ingested, operators will need to know which hints are in place (and, later, add new ones).

## Goal

Surface all existing schema hints in the admin UI as a **read-only** page, grouped by scope, so an operator can see what hints exist, their type, target column, and mappings. Structured so add/edit can be layered on later.

## Non-Goals

- No add / edit / delete (read-only v1; write paths deferred). Hints continue to be created by seed scripts for now.
- No "which registered tables each hint resolves onto" (would require cross-referencing the schema registry; deferred).
- No change to how hints are resolved or used in text-to-SQL.

## Background (existing pieces this reuses)

- `MetadataStore.load_all_hints()` → `list[SchemaHint]` (already exists, `src/db/metadata.py`).
- `SchemaHint` (`src/db/hint_store.py`): `scope_type` (`category`|`dataset`), `scope_value` (category name, or dataset id as string), `hint_type` (`value_glossary`|`column_note`|`table_note`), `target_column` (None for table_note), `payload` (dict: `{code: meaning}` for glossary, `{"text": ...}` for notes), `provenance` (`curated`|`auto`|`learned`), `confidence`, `id`, `created_at`, `created_by`.
- `MetadataStore.list_datasets(active_only=False)` → datasets with `.id` and `.name` (to resolve dataset-scoped hints to a readable name).
- Admin page pattern (`src/admin/routes.py`): `@router.get(...)` → `_require_login(request)` → `store = get_metadata_store()` → load data → `templates.TemplateResponse(request, "<page>.html", {...})`. Templates extend `base.html`; the left nav lives in `base.html`.

## Design

### Components

1. **`_build_hints_view(hints, datasets) -> list[dict]`** — a pure helper (in `src/admin/routes.py`) that turns the flat hint list into a grouped, template-ready view-model. The one unit-tested unit.
   - Group hints by scope. Scope label: `category = <scope_value>` for category scope; `dataset = <name> (id <scope_value>)` for dataset scope, resolving `<name>` from the datasets list (fallback to `dataset = <scope_value>` if the id isn't found).
   - Sort groups: category scopes first then dataset scopes, each alphabetically by label.
   - Within a group, sort hints by type order `table_note`, `column_note`, `value_glossary`, then by `target_column`.
   - Per hint, produce a dict:
     - `hint_type`, `target_column`, `provenance`, `confidence`.
     - For `value_glossary`: `entries` = sorted list of `{"code": k, "meaning": v}` from payload, and `count` = len(entries).
     - For `column_note`/`table_note`: `text` = `payload.get("text", "")`.
   - Returns: `[{"scope_label": str, "hints": [ ...hint dicts... ]}, ...]`.

2. **Route `GET /admin/hints`** (`src/admin/routes.py`):
   ```python
   @router.get("/hints", response_class=HTMLResponse)
   async def hints_page(request: Request):
       redirect = _require_login(request)
       if redirect:
           return redirect
       store = get_metadata_store()
       hints = await store.load_all_hints()
       datasets = await store.list_datasets(active_only=False)
       groups = _build_hints_view(hints, datasets)
       return templates.TemplateResponse(request, "hints.html", {"groups": groups})
   ```

3. **Template `src/admin/templates/hints.html`** — extends `base.html`. Iterates `groups`. For each group, render the scope label as a heading, then each hint:
   - `value_glossary` → a `<details>` whose `<summary>` shows `column "<target_column>" · <provenance> · <count> values`, expanding to a small two-column list of `code → meaning`.
   - `column_note`/`table_note` → the note text (with target column label for column notes).
   - Empty state when `groups` is empty: "No schema hints yet — they're added by seed scripts (e.g. `scripts/seed_military_paygrade_glossary.py`)."

4. **Nav link in `src/admin/templates/base.html`** — add `<a href="/admin/hints">Schema Hints</a>` near the other domain links (e.g. after Knowledge Graph, before Audit Log).

### Data flow

```
GET /admin/hints
  -> _require_login
  -> store.load_all_hints()  +  store.list_datasets()
  -> _build_hints_view(hints, datasets)  -> grouped view-model
  -> render hints.html (read-only)
```

### Display (mockup)

```
Schema Hints                                    (read-only)

category = payroll_compensation
   value_glossary · column "locname" · curated      [▸ 55 values]
        AK  → State of Alaska
        MFL → Miami-Fort Lauderdale-Port St. Lucie, FL
        RUS → Rest of U.S.
   column_note · column "locname"
        "OPM locality pay area code. Match the place the user names to its code…"

dataset = Military Pay (id 2)
   value_glossary · column "col_0" · curated         [▸ 12 values]
        O-1E → Commissioned Officer with prior enlisted service
   table_note
        "U.S. military active-duty monthly basic pay; rows are pay grades…"
```

## Error handling

Read-only and additive. `_build_hints_view` tolerates a glossary payload that isn't a dict (treat as no entries) and a missing `text` key (empty string), so a malformed hint renders blank rather than 500-ing the page. The route follows the existing login-gate + `get_metadata_store()` pattern; a metadata error surfaces the same way as other admin pages.

## Testing

- **Unit — `_build_hints_view` (primary):** given a `value_glossary` (`{"AK": "State of Alaska", "MFL": "Miami..."}` on `category=payroll_compensation`) + a `column_note` (same scope) + a `table_note` on `dataset=2`, and a datasets list mapping id 2 → "Military Pay", assert: two groups; category group sorts before dataset group; dataset label resolves to `dataset = Military Pay (id 2)`; the glossary hint dict has `count == 2` and sorted `entries`; the note hint has the right `text`; missing-dataset id falls back to `dataset = <id>`.
- **Route smoke:** `GET /admin/hints` (authenticated session) returns 200 and the body contains a scope label, using a stubbed/real metadata store. Confirms the route wires up and the template renders.

## Risks / considerations

- Long glossaries (55+ entries) are collapsed in `<details>` so the page stays scannable; default collapsed.
- v1 is read-only; the grouped structure intentionally leaves room for a later "Add hint" affordance without restructuring.
