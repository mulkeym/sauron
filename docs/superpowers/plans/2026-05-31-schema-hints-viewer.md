# Schema Hints Viewer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only `/admin/hints` page that lists all schema hints (value glossaries + column/table notes) grouped by scope.

**Architecture:** A pure `_build_hints_view` helper groups `load_all_hints()` output into a template-ready view-model; a login-gated `GET /admin/hints` route renders it via a new `hints.html` template; a nav link is added to `base.html`. Read-only — no write paths (the existing `/api/hints` write routes are untouched).

**Tech Stack:** FastAPI, Jinja2 templates, pytest (+ FastAPI TestClient).

**Spec:** `docs/superpowers/specs/2026-05-31-schema-hints-viewer-design.md`

---

## File Structure

- `src/admin/routes.py` — add pure helper `_build_hints_view` + `GET /admin/hints` route (modify).
- `src/admin/templates/hints.html` — new template extending `base.html` (create).
- `src/admin/templates/base.html` — add the nav link (modify).
- `tests/test_admin/test_hint_routes.py` — unit tests for `_build_hints_view` + route smoke test (modify; file already exists).

Context already in place (reuse, do not recreate): `MetadataStore.load_all_hints()`, `MetadataStore.list_datasets(active_only=False)`, `SchemaHint` (`src/db/hint_store.py`), the `_require_login` + `get_metadata_store()` + `templates.TemplateResponse(request, name, ctx)` admin pattern, and `get_metadata_store` imported into `src.admin.routes`.

---

## Task 1: `_build_hints_view` helper

**Files:**
- Modify: `src/admin/routes.py` (add helper after `list_hints_route`, ~line 548)
- Test: `tests/test_admin/test_hint_routes.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_admin/test_hint_routes.py`:

```python
from types import SimpleNamespace
from src.db.hint_store import SchemaHint
from src.admin.routes import _build_hints_view


def _hints():
    return [
        SchemaHint(scope_type="category", scope_value="payroll_compensation",
                   hint_type="value_glossary", target_column="locname",
                   payload={"MFL": "Miami", "AK": "State of Alaska"}),
        SchemaHint(scope_type="category", scope_value="payroll_compensation",
                   hint_type="column_note", target_column="locname",
                   payload={"text": "OPM locality code"}),
        SchemaHint(scope_type="dataset", scope_value="2",
                   hint_type="table_note", target_column=None,
                   payload={"text": "Military pay"}),
    ]


def test_build_hints_view_groups_and_labels():
    datasets = [SimpleNamespace(id=2, name="Military Pay")]
    groups = _build_hints_view(_hints(), datasets)
    labels = [g["scope_label"] for g in groups]
    # category group sorts before dataset group; dataset id -> name resolved
    assert labels == ["category = payroll_compensation", "dataset = Military Pay (id 2)"]


def test_build_hints_view_glossary_entries_sorted_and_counted():
    groups = _build_hints_view(_hints(), [SimpleNamespace(id=2, name="Military Pay")])
    cat = next(g for g in groups if g["scope_label"].startswith("category"))
    # within a group: table_note(none here), column_note before value_glossary
    types = [h["hint_type"] for h in cat["hints"]]
    assert types == ["column_note", "value_glossary"]
    gloss = next(h for h in cat["hints"] if h["hint_type"] == "value_glossary")
    assert gloss["count"] == 2
    assert gloss["entries"] == [{"code": "AK", "meaning": "State of Alaska"},
                                {"code": "MFL", "meaning": "Miami"}]
    note = next(h for h in cat["hints"] if h["hint_type"] == "column_note")
    assert note["text"] == "OPM locality code"


def test_build_hints_view_unknown_dataset_falls_back_to_id():
    h = [SchemaHint(scope_type="dataset", scope_value="9", hint_type="table_note",
                    target_column=None, payload={"text": "x"})]
    groups = _build_hints_view(h, [])  # no datasets -> no name
    assert groups[0]["scope_label"] == "dataset = 9"
```

- [ ] **Step 2: Run to confirm fail**

Run: `python3 -m pytest tests/test_admin/test_hint_routes.py -k build_hints_view -v`
Expected: FAIL with `ImportError: cannot import name '_build_hints_view'`.

- [ ] **Step 3: Implement the helper**

In `src/admin/routes.py`, add this function immediately after the `list_hints_route` function (the one ending around line 547):

```python
def _build_hints_view(hints, datasets):
    """Group SchemaHints by scope into a template-ready view-model. Pure/sync.
    Returns [{"scope_label": str, "hints": [hint-dict, ...]}, ...] with category
    scopes before dataset scopes. hint-dict carries hint_type/target_column/
    provenance/confidence, plus either entries+count (value_glossary) or text (notes)."""
    ds_name = {str(d.id): d.name for d in datasets}

    def scope_label(h):
        if h.scope_type == "dataset":
            name = ds_name.get(str(h.scope_value))
            return f"dataset = {name} (id {h.scope_value})" if name else f"dataset = {h.scope_value}"
        return f"category = {h.scope_value}"

    type_order = {"table_note": 0, "column_note": 1, "value_glossary": 2}
    groups: dict[str, list] = {}
    for h in hints:
        groups.setdefault(scope_label(h), []).append(h)

    out = []
    for label in sorted(groups, key=lambda l: (0 if l.startswith("category") else 1, l.lower())):
        hint_dicts = []
        for h in sorted(groups[label], key=lambda h: (type_order.get(h.hint_type, 9), h.target_column or "")):
            d = {"hint_type": h.hint_type, "target_column": h.target_column,
                 "provenance": h.provenance, "confidence": h.confidence}
            if h.hint_type == "value_glossary":
                payload = h.payload if isinstance(h.payload, dict) else {}
                d["entries"] = [{"code": k, "meaning": payload[k]} for k in sorted(payload)]
                d["count"] = len(d["entries"])
            else:
                d["text"] = h.payload.get("text", "") if isinstance(h.payload, dict) else ""
            hint_dicts.append(d)
        out.append({"scope_label": label, "hints": hint_dicts})
    return out
```

- [ ] **Step 4: Run to confirm pass**

Run: `python3 -m pytest tests/test_admin/test_hint_routes.py -k build_hints_view -v`
Expected: 3 passed.

- [ ] **Step 5: Run the whole hint-routes file (no regression on existing hint tests)**

Run: `python3 -m pytest tests/test_admin/test_hint_routes.py -q`
Expected: all pass (the 3 new + the pre-existing create/list/bulk tests).

- [ ] **Step 6: Commit**

```bash
git add src/admin/routes.py tests/test_admin/test_hint_routes.py
git commit -m "feat: _build_hints_view helper to group schema hints by scope"
```

---

## Task 2: `/admin/hints` page (route + template + nav)

**Files:**
- Modify: `src/admin/routes.py` (add `GET /hints` route after `_build_hints_view`)
- Create: `src/admin/templates/hints.html`
- Modify: `src/admin/templates/base.html` (nav link)
- Test: `tests/test_admin/test_hint_routes.py`

- [ ] **Step 1: Write the failing route smoke test**

Append to `tests/test_admin/test_hint_routes.py` (mirrors the working `test_dashboard_loads` in test_routes.py — patches `src.admin.routes.get_metadata_store`):

```python
def test_hints_page_renders():
    from unittest.mock import patch, AsyncMock
    from fastapi.testclient import TestClient
    from src.main import create_app
    client = TestClient(create_app())
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        store.load_all_hints.return_value = [
            SchemaHint(scope_type="category", scope_value="payroll_compensation",
                       hint_type="value_glossary", target_column="locname",
                       payload={"AK": "State of Alaska"}),
        ]
        store.list_datasets.return_value = []
        mock_get.return_value = store
        resp = client.get("/admin/hints")
    assert resp.status_code == 200
    assert "payroll_compensation" in resp.text
    assert "State of Alaska" in resp.text
```

- [ ] **Step 2: Run to confirm fail**

Run: `python3 -m pytest tests/test_admin/test_hint_routes.py -k hints_page_renders -v`
Expected: FAIL (404 — no `/admin/hints` route yet — so `resp.status_code == 200` fails, or template missing).

- [ ] **Step 3: Add the route**

In `src/admin/routes.py`, immediately after the `_build_hints_view` helper (from Task 1), add:

```python
@router.get("/hints", response_class=HTMLResponse)
async def hints_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    store = get_metadata_store()
    hints = await store.load_all_hints()
    datasets = await store.list_datasets(active_only=False)
    return templates.TemplateResponse(request, "hints.html",
                                      {"groups": _build_hints_view(hints, datasets)})
```

- [ ] **Step 4: Create the template**

Create `src/admin/templates/hints.html`:

```html
{% extends "base.html" %}
{% block title %}Schema Hints - SAURON{% endblock %}
{% block content %}
<h1>Schema Hints <span style="font-size:0.8rem;color:#888;">(read-only)</span></h1>
<p class="section-desc">Domain knowledge injected into text-to-SQL: value glossaries (code &rarr; meaning) and column/table notes, grouped by scope.</p>

{% if not groups %}
<p>No schema hints yet &mdash; they're added by seed scripts (e.g. <code>scripts/seed_military_paygrade_glossary.py</code>).</p>
{% else %}
{% for g in groups %}
<div class="settings-section">
  <h2>{{ g.scope_label }}</h2>
  {% for h in g.hints %}
    {% if h.hint_type == "value_glossary" %}
    <details>
      <summary>value_glossary &middot; column "{{ h.target_column }}" &middot; {{ h.provenance }} &middot; {{ h.count }} values</summary>
      <table style="margin:0.5rem 0;">
        {% for e in h.entries %}<tr><td style="padding-right:1rem;"><code>{{ e.code }}</code></td><td>&rarr; {{ e.meaning }}</td></tr>{% endfor %}
      </table>
    </details>
    {% else %}
    <p><strong>{{ h.hint_type }}</strong>{% if h.target_column %} &middot; column "{{ h.target_column }}"{% endif %} &middot; {{ h.provenance }}<br>
       <span style="color:#555;">{{ h.text }}</span></p>
    {% endif %}
  {% endfor %}
</div>
{% endfor %}
{% endif %}
{% endblock %}
```

- [ ] **Step 5: Add the nav link**

In `src/admin/templates/base.html`, the nav currently has these consecutive lines:

```html
            <a href="/admin/knowledge-graph">Knowledge Graph</a>
            <a href="/admin/audit">Audit Log</a>
```

Insert the hints link between them so it reads:

```html
            <a href="/admin/knowledge-graph">Knowledge Graph</a>
            <a href="/admin/hints">Schema Hints</a>
            <a href="/admin/audit">Audit Log</a>
```

- [ ] **Step 6: Run the route smoke test + full hint-routes file**

Run: `python3 -m pytest tests/test_admin/test_hint_routes.py -v`
Expected: all pass (helper tests + the new route smoke test + pre-existing hint tests).

- [ ] **Step 7: Run the admin route suite (no regression)**

Run: `python3 -m pytest tests/test_admin/ -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/admin/routes.py src/admin/templates/hints.html src/admin/templates/base.html tests/test_admin/test_hint_routes.py
git commit -m "feat: read-only Schema Hints admin page (/admin/hints) + nav link"
```

---

## Task 3: Full suite + live smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the admin + agent suites**

Run: `python3 -m pytest tests/test_admin/ tests/test_agent/ -q`
Expected: all pass (no new failures).

- [ ] **Step 2: Live smoke (post-deploy)**

After rebuild + recreate of `sauron-api-1`, confirm the page renders the real seeded hints:

```bash
docker exec sauron-api-1 python3 -c "
import asyncio
from src.api.routes_ingest import get_metadata_store, get_hint_store
from src.ingestion.tabular_ingest import populate_hint_store
from src.admin.routes import _build_hints_view
store = get_metadata_store()
asyncio.run(populate_hint_store(store, get_hint_store()))
hints = asyncio.run(store.load_all_hints())
datasets = asyncio.run(store.list_datasets(active_only=False))
groups = _build_hints_view(hints, datasets)
for g in groups:
    print(g['scope_label'])
    for h in g['hints']:
        extra = (str(h['count']) + ' values') if h['hint_type']=='value_glossary' else repr(h.get('text','')[:50])
        print('   ', h['hint_type'], h.get('target_column'), '->', extra)
"
```

Expected: prints the `category = payroll_compensation` glossary (locname, ~55 values) + column_note, and the `dataset = ...` military pay glossary + table_note. (Confirms the page would render the live hints.) Also browse to `/admin/hints` in the UI to eyeball it.

- [ ] **Step 3: Final status**

```bash
git status   # clean
```

---

## Self-Review Notes (completed by plan author)

- **Spec coverage:** `_build_hints_view` grouping/label/sort/payload-summary (Task 1, matches spec §Components.1); `GET /admin/hints` login-gated route (Task 2, spec §Components.2); `hints.html` with collapsible glossaries + notes + empty state (Task 2, spec §Components.3 + Display); nav link (Task 2, spec §Components.4); error tolerance for non-dict payload / missing text (helper `isinstance` guards, spec §Error handling); unit test of helper + route smoke (Tasks 1 & 2, spec §Testing). Read-only, no write paths (spec §Non-Goals). No gaps.
- **Type/signature consistency:** `_build_hints_view(hints, datasets)` defined in Task 1, called identically in the Task 2 route and tests. View-model keys (`scope_label`, `hints`, `hint_type`, `target_column`, `provenance`, `count`, `entries`{`code`,`meaning`}, `text`) match between the helper, the template, and the tests.
- **Placeholder scan:** none — full code in every step; the live-smoke is a real diagnostic.
