# Settings Shell Reorganization

**Date:** 2026-05-31
**Status:** Design approved, pending spec review

## Problem

The admin top nav has grown to 11+ links. Several are management/data pages (Categories, Proposals, Connectors, Schema Hints, Audit Log) that clutter the top bar. The Settings page is one long scrolling page with a single save-form spanning Security/Models/Retrieval/System and a Maintenance section. There's no organizing structure.

## Goal

Reorganize the admin UI so the operational/management pages and configuration sections live under **Settings**, presented via a **left sub-nav** ("settings shell"). Shrink the top bar to the high-traffic pages. Keep all existing functionality.

## Non-Goals

- No functional change to Categories/Proposals/Connectors/Schema Hints/Audit logic — only re-homing their templates into the shell.
- No new settings/config fields.
- No URL move for the management pages (kept at their current routes to avoid redirects/link churn).
- Not addressing the pre-existing `test_routes.py` auth-pattern failures (separate cleanup).

## Design

### 1. Settings shell

New template `src/admin/templates/settings_base.html` extending `base.html`. It renders a two-pane layout: a **left sub-nav** + a `{% block settings_content %}` content area. Every settings section template extends `settings_base.html` and fills `settings_content`. The active sub-nav item is highlighted from a context variable `active` (a short key like `"models"`, `"categories"`).

The left sub-nav (two visually-separated groups, in this order):

```
CONFIGURATION                MANAGEMENT
  Security      security       Categories     categories
  Models        models         Proposals      proposals
  Retrieval &   retrieval      Connectors     connectors
   Processing                  Schema Hints   hints
  System        system         Audit Logs     audit
  Maintenance   maintenance
```

(Right column shows the `active` key used per item.)

Nav item links:
- Configuration items → `/admin/settings/<key>` (new routes).
- Management items → their existing routes: Categories `/admin/categories`, Proposals `/admin/proposals`, Connectors `/admin/connectors`, Schema Hints `/admin/hints`, Audit Logs `/admin/audit`.

### 2. Top-bar cleanup

In `src/admin/templates/base.html` `.nav-links`, remove the five links: Categories, Proposals, Connectors, Schema Hints, Audit Log. Resulting top nav: Dashboard, Datasets, Documents, Queue, Playground, Knowledge Graph, Settings, Logout (+ theme toggle). The Settings link stays `/admin/settings`.

### 3. Configuration split + per-section save

The current single settings form (in `settings.html`) splits into five section templates, each extending `settings_base.html`, each with **its own form + Save button** posting only that section's fields:

- `settings_security.html` (active `security`) — admin username/password fields.
- `settings_models.html` (active `models`) — LLM + embedding model config.
- `settings_retrieval.html` (active `retrieval`) — the current "Retrieval & Processing" fields.
- `settings_system.html` (active `system`) — the current "System" fields.
- `settings_maintenance.html` (active `maintenance`) — the Maintenance sub-sections (backup/restore/etc.), which already use their own HTMX endpoints, not the settings form.

Routes (all login-gated, in `src/admin/routes.py`):
- `GET /admin/settings` → redirect to `/admin/settings/security`.
- `GET /admin/settings/security|models|retrieval|system|maintenance` → render the matching template with `active` set and `{"settings": settings}` in context.

**Partial save (backend, the one real risk):** `POST /api/settings` currently overwrites every persisted field. Change it to a **partial merge**: read the existing persisted config (`data/settings.json` if present, else current `settings` values), update only the keys present in the submitted form, apply to the live `settings` object, and write the merged dict back to `data/settings.json`. So saving Models never clears Security. Form fields keep their existing `name=` attributes so the handler maps them the same way; it just iterates submitted keys instead of assuming all are present.

### 4. Management re-homing

Categories/Proposals/Connectors/Schema Hints/Audit keep their existing routes and route handlers unchanged except for one added context key. Their templates change to **extend `settings_base.html`** (instead of `base.html`) and move their content into `{% block settings_content %}`. Their HTMX endpoints, forms, and JS are untouched.

Highlighting is uniform: `settings_base.html` reads a context variable `active` (a short key string) to mark the active sub-nav item. Every settings route — config and management alike — passes `"active": "<key>"` in its `TemplateResponse` context (a one-line addition per management route; set directly for the config routes).

### Data flow

```
GET /admin/settings            -> redirect /admin/settings/security
GET /admin/settings/<key>      -> settings_<key>.html (extends settings_base, active=<key>)
GET /admin/categories|...      -> existing handler + {"active": "<key>"} -> template now extends settings_base
POST /admin/api/settings       -> partial-merge submitted fields into data/settings.json + live settings
```

## Error handling

- Partial save: unknown/absent fields are simply not updated (merge semantics), so a section's save can't wipe another's values. Persisting to `data/settings.json` follows the existing write path.
- An unknown `/admin/settings/<key>` returns 404 (only the five known keys are routed).
- Login gating is unchanged (each route calls `_require_login` as the existing pages do).

## Testing

- **Shell:** `settings_base.html` renders the full sub-nav; given `active="models"`, the Models item carries the active marker/class and others don't (assert via a rendered route).
- **Config section routes:** `GET /admin/settings/security` (and models/retrieval/system/maintenance) return 200 and contain that section's field(s) + a Save control. `GET /admin/settings` 30x-redirects to `/admin/settings/security`.
- **Partial save (critical):** start from a known `data/settings.json` with both a Security value and a System value; POST only Models fields to `/admin/api/settings`; assert the Models value changed AND the Security and System values are unchanged in the persisted file. A second test: posting Security fields leaves Models/System untouched.
- **Management re-home:** `GET /admin/hints` (and one other, e.g. `/admin/categories`) now renders inside the shell — response contains a sub-nav marker (e.g. a link to `/admin/settings/security`) confirming `settings_base.html` is in use.
- **Top nav:** `base.html`-rendered pages no longer contain `/admin/categories` etc. in the top `.nav-links`, but still contain `/admin/settings`. (Note: those management links now appear in the settings sub-nav, so assert against the top `nav-links` region specifically, or assert the count/known-removed text in a page that doesn't include the sub-nav, e.g. Dashboard.)

Tests use the existing admin test pattern (FastAPI `TestClient` + `patch("src.admin.routes._is_authenticated", return_value=True)` + patched `get_metadata_store`).

## Risks / considerations

- **Partial-save backend** is the highest-risk change; covered by dedicated tests above. If any setting field is a checkbox (absent when unchecked), partial-merge must not silently drop it — each config section's form should include all of its own boolean fields explicitly (hidden default or always-posted), so "absent" means "not this section" rather than "unchecked". The implementation must verify each split section's boolean fields round-trip.
- **URL inconsistency** (config under `/admin/settings/*`, management at top-level routes) is accepted for low churn.
- Re-homing templates must preserve each page's existing `{% block content %}` markup verbatim inside the new `{% block settings_content %}` so no functionality/HTMX wiring is lost.
