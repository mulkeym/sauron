# Settings Shell Reorganization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the admin UI so configuration sections and the management pages (Categories/Proposals/Connectors/Schema Hints/Audit) live under a Settings shell with a left sub-nav, and the top nav shrinks.

**Architecture:** A `settings_base.html` shell (left sub-nav + content block) that section templates extend. The single config form splits into per-section pages with per-section save, backed by a partial-merge `_apply_settings_update` helper (the one unit-tested, risky piece). Management pages keep their routes but re-skin into the shell.

**Tech Stack:** FastAPI, Jinja2, htmx, pytest + FastAPI TestClient.

**Spec:** `docs/superpowers/specs/2026-05-31-settings-shell-reorganization-design.md`

---

## File Structure

- `src/admin/routes.py` — `_SETTINGS_FIELDS`/`_SETTINGS_KEEP_IF_BLANK` + `_apply_settings_update` helper; rewrite `save_settings`; `/admin/settings` redirect + 5 config section routes; `active` added to 5 management routes (modify).
- `src/admin/templates/settings_base.html` — the shell (create).
- `src/admin/templates/settings_security.html|settings_models.html|settings_retrieval.html|settings_system.html|settings_maintenance.html` — split config sections (create).
- `src/admin/templates/settings.html` — delete (replaced by the split sections).
- `src/admin/templates/categories.html|proposals.html|connectors.html|hints.html|audit.html` — re-skin to extend the shell (modify).
- `src/admin/templates/base.html` — remove 5 top-nav links (modify).
- `tests/test_admin/test_settings_shell.py` — new test file (create).

Existing facts: `settings.html` main form is `<form hx-post="/admin/api/settings">` (line 6) wrapping Security (9–42), Models (43–109), Retrieval & Processing (110–206), System (207–236), with one Save (237) and `</form>` (240). Maintenance (244–313) is OUTSIDE the form (independent htmx action buttons + a restore form). Admin tests use `TestClient(create_app())` + `patch("src.admin.routes._is_authenticated", return_value=True)`.

---

## Task 1: Partial-save backend (`_apply_settings_update`)

**Files:**
- Modify: `src/admin/routes.py` (replace `save_settings`, lines 1882–1971; add helper + field tables above it)
- Test: `tests/test_admin/test_settings_shell.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_admin/test_settings_shell.py`:

```python
from src.admin.routes import _apply_settings_update
from src.config import settings


def test_apply_settings_update_is_partial():
    settings.admin_username = "ADMIN_KEEP"
    settings.llm_max_context = 250000
    settings.vllm_model_name = "OLD"
    persist = _apply_settings_update({"vllm_model_name": "NEW", "vllm_base_url": "http://x"})
    assert settings.vllm_model_name == "NEW"          # submitted -> updated
    assert settings.admin_username == "ADMIN_KEEP"     # absent -> untouched
    assert settings.llm_max_context == 250000          # absent -> untouched
    assert persist["vllm_model_name"] == "NEW"
    assert persist["admin_username"] == "ADMIN_KEEP"
    assert persist["llm_max_context"] == 250000


def test_apply_settings_update_bool_roundtrip():
    settings.feedback_enabled = True
    _apply_settings_update({"feedback_enabled": "false"})   # unchecked box posts hidden "false"
    assert settings.feedback_enabled is False
    _apply_settings_update({"feedback_enabled": "true"})
    assert settings.feedback_enabled is True


def test_apply_settings_update_keeps_blank_credential():
    settings.admin_password = "secret"
    _apply_settings_update({"admin_password": ""})           # blank submit must not clear it
    assert settings.admin_password == "secret"


def test_apply_settings_update_casts_numbers():
    _apply_settings_update({"llm_max_context": "300000", "feedback_similarity_threshold": "0.7"})
    assert settings.llm_max_context == 300000 and isinstance(settings.llm_max_context, int)
    assert settings.feedback_similarity_threshold == 0.7
```

- [ ] **Step 2: Run to confirm fail**

Run: `python3 -m pytest tests/test_admin/test_settings_shell.py -k apply_settings -v`
Expected: FAIL with `ImportError: cannot import name '_apply_settings_update'`.

- [ ] **Step 3: Implement the helper + field tables**

In `src/admin/routes.py`, immediately BEFORE the `save_settings` route (line ~1882), add:

```python
# Form field -> caster. Membership mirrors the persisted settings dict.
_SETTINGS_FIELDS = {
    "admin_username": str, "admin_password": str, "api_keys": str,
    "vllm_base_url": str, "vllm_model_name": str, "vllm_api_key": str,
    "embedding_mode": str, "embedding_api_url": str, "embedding_model_name": str,
    "mcp_port": int, "mcp_alt_port": int,
    "entity_merge_auto_threshold": float, "entity_merge_review_threshold": float,
    "max_parallel_ingestion": int, "llm_concurrency": int,
    "llm_max_context": int, "llm_max_output_tokens": int, "metadata_max_doc_length": int,
    "metadata_extraction_enabled": bool, "feedback_enabled": bool,
    "prf_enabled": bool, "strategy_memory_enabled": bool,
    "feedback_similarity_threshold": float,
}
# String fields that must NOT be cleared when submitted blank (creds/urls). vllm_api_key may be blanked.
_SETTINGS_KEEP_IF_BLANK = {
    "admin_username", "admin_password", "api_keys",
    "vllm_base_url", "vllm_model_name",
    "embedding_mode", "embedding_api_url", "embedding_model_name",
}


def _apply_settings_update(form) -> dict:
    """Partial update of the live `settings` object from a submitted form: only
    fields PRESENT in the form are touched (so a per-section save never clobbers
    another section). Returns the full persist dict. ``form`` is a Starlette
    FormData (has .getlist) or a plain dict (tests). Booleans use the last value
    (sections post a hidden 'false' + checkbox 'true', so an unchecked box still
    posts 'false')."""
    def last(name):
        return form.getlist(name)[-1] if hasattr(form, "getlist") else form[name]

    for name, caster in _SETTINGS_FIELDS.items():
        if name not in form:
            continue
        raw = last(name)
        if caster is bool:
            val = str(raw).strip().lower() in ("true", "1", "on", "yes")
        else:
            s = str(raw).strip()
            if s == "" and name in _SETTINGS_KEEP_IF_BLANK:
                continue
            val = caster(s) if s != "" else getattr(settings, name)
        setattr(settings, name, val)

    return {name: getattr(settings, name) for name in _SETTINGS_FIELDS}
```

Then REPLACE the entire existing `save_settings` function (the `@router.post("/api/settings")` decorator + the `async def save_settings(...)` with all its `Form(...)` params, the body, the `persist = {...}` dict, the write, and the return — lines ~1882 through 1971) with:

```python
@router.post("/api/settings")
async def save_settings(request: Request):
    """Persist a partial settings update (only the submitted section's fields)."""
    persist = _apply_settings_update(await request.form())
    Path("data/settings.json").write_text(json.dumps(persist, indent=2) + "\n")
    return HTMLResponse('<div class="status-ok">Settings saved successfully.</div>')
```

(`Request`, `HTMLResponse`, `Path`, `json`, `settings`, `Form` are all already imported in this module; `Form` is still used by other routes, leave its import.)

- [ ] **Step 4: Run to confirm pass**

Run: `python3 -m pytest tests/test_admin/test_settings_shell.py -k apply_settings -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/admin/routes.py tests/test_admin/test_settings_shell.py
git commit -m "feat: partial settings save (_apply_settings_update) so per-section save won't clobber others"
```

---

## Task 2: Settings shell + config section split

**Files:**
- Create: `src/admin/templates/settings_base.html`, `settings_security.html`, `settings_models.html`, `settings_retrieval.html`, `settings_system.html`, `settings_maintenance.html`
- Modify: `src/admin/routes.py` (`settings_page` → redirect; add 5 section routes)
- Delete: `src/admin/templates/settings.html`
- Test: `tests/test_admin/test_settings_shell.py`

- [ ] **Step 1: Write the failing route tests**

Append to `tests/test_admin/test_settings_shell.py`:

```python
def _client():
    from fastapi.testclient import TestClient
    from src.main import create_app
    return TestClient(create_app())


def test_settings_redirects_to_first_section():
    from unittest.mock import patch
    with patch("src.admin.routes._is_authenticated", return_value=True):
        resp = _client().get("/admin/settings", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"].endswith("/admin/settings/security")


def test_each_config_section_renders_with_active_marker():
    from unittest.mock import patch
    c = _client()
    for key in ["security", "models", "retrieval", "system", "maintenance"]:
        with patch("src.admin.routes._is_authenticated", return_value=True):
            resp = c.get(f"/admin/settings/{key}")
        assert resp.status_code == 200, key
        # the shell sub-nav links to every section; the active one carries the marker class
        assert 'class="subnav-item active"' in resp.text, key
        # active item must be THIS key's link
        assert f'href="/admin/settings/{key}"' in resp.text or key in ("maintenance",), key
```

- [ ] **Step 2: Run to confirm fail**

Run: `python3 -m pytest tests/test_admin/test_settings_shell.py -k "section or redirect" -v`
Expected: FAIL (routes/templates don't exist yet → 404 / assertion failures).

- [ ] **Step 3: Create the shell `settings_base.html`**

Create `src/admin/templates/settings_base.html`:

```html
{% extends "base.html" %}
{% set active = active | default('') %}
{% block content %}
<div class="settings-shell" style="display:flex; gap:1.5rem; align-items:flex-start;">
  <aside class="settings-subnav" style="min-width:190px; flex:0 0 auto;">
    {% set groups = [
      ("Configuration", [
        ("security","Security","/admin/settings/security"),
        ("models","Models","/admin/settings/models"),
        ("retrieval","Retrieval & Processing","/admin/settings/retrieval"),
        ("system","System","/admin/settings/system"),
        ("maintenance","Maintenance","/admin/settings/maintenance")]),
      ("Management", [
        ("categories","Categories","/admin/categories"),
        ("proposals","Proposals","/admin/proposals"),
        ("connectors","Connectors","/admin/connectors"),
        ("hints","Schema Hints","/admin/hints"),
        ("audit","Audit Logs","/admin/audit")]),
    ] %}
    {% for group_name, items in groups %}
    <div class="subnav-group" style="margin-bottom:1rem;">
      <div class="subnav-group-title" style="font-size:0.7rem; text-transform:uppercase; color:#888; letter-spacing:0.05em; margin:0 0 0.25rem 0.25rem;">{{ group_name }}</div>
      {% for key, label, href in items %}
      <a class="subnav-item{% if key == active %} active{% endif %}" href="{{ href }}"
         style="display:block; padding:0.35rem 0.6rem; border-radius:6px; text-decoration:none;{% if key == active %} font-weight:600; background:rgba(120,120,160,0.18);{% endif %}">{{ label }}</a>
      {% endfor %}
    </div>
    {% endfor %}
  </aside>
  <section class="settings-pane" style="flex:1; min-width:0;">
    {% block settings_content %}{% endblock %}
  </section>
</div>
{% endblock %}
```

- [ ] **Step 4: Create the 5 config section templates (verbatim content from settings.html)**

For each section below: create the file, with this skeleton, pasting the indicated `<div class="settings-section">…</div>` block from the CURRENT `src/admin/templates/settings.html` VERBATIM inside the form, then DELETE that block from settings.html at the end of Step 6.

`settings_security.html` (content = settings.html lines 9–42):
```html
{% extends "settings_base.html" %}
{% block title %}Security - Settings{% endblock %}
{% block settings_content %}
<h1>Security</h1>
<form hx-post="/admin/api/settings" hx-target="#save-status" hx-swap="innerHTML">
  <!-- PASTE settings.html lines 9-42 (the <div class="settings-section"><h2>Security</h2>...</div>) here verbatim -->
  <button type="submit">Save</button>
  <span id="save-status" style="margin-left:0.5rem;"></span>
</form>
{% endblock %}
```

`settings_models.html` (content = settings.html lines 43–109): same skeleton, `<h1>Models</h1>`, paste the Models `<div class="settings-section">…</div>` verbatim.

`settings_retrieval.html` (content = settings.html lines 110–206): same skeleton, `<h1>Retrieval &amp; Processing</h1>`, paste the Retrieval block verbatim. **Important:** this section contains boolean checkboxes (`feedback_enabled`, `prf_enabled`, `strategy_memory_enabled`, and possibly `metadata_extraction_enabled`). For EACH checkbox `<input type="checkbox" name="X" ...>` in this section, insert immediately BEFORE it a hidden default so an unchecked box still posts a value:
```html
<input type="hidden" name="X" value="false">
```
(The partial-save handler takes the last value, so checked → "true" wins, unchecked → only the hidden "false" posts.)

`settings_system.html` (content = settings.html lines 207–236): same skeleton, `<h1>System</h1>`, paste the System block verbatim; apply the same hidden-checkbox treatment to any boolean checkboxes it contains.

`settings_maintenance.html` (content = settings.html lines 244–313, the Maintenance block — NO save form, these are independent action buttons):
```html
{% extends "settings_base.html" %}
{% block title %}Maintenance - Settings{% endblock %}
{% block settings_content %}
<h1>Maintenance</h1>
<!-- PASTE settings.html lines 244-313 (the <h2>Maintenance</h2> through the restore </form>) here verbatim -->
{% endblock %}
```

- [ ] **Step 5: Replace the settings route + add section routes**

In `src/admin/routes.py`, replace the existing `settings_page` function:

```python
@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    return templates.TemplateResponse(request, "settings.html", {"settings": settings})
```

with:

```python
from fastapi.responses import RedirectResponse  # if not already imported at top; safe to import here

@router.get("/settings")
async def settings_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    return RedirectResponse(url="/admin/settings/security", status_code=307)


@router.get("/settings/{section}", response_class=HTMLResponse)
async def settings_section_page(request: Request, section: str):
    redirect = _require_login(request)
    if redirect:
        return redirect
    if section not in ("security", "models", "retrieval", "system", "maintenance"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404)
    return templates.TemplateResponse(request, f"settings_{section}.html",
                                      {"settings": settings, "active": section})
```

(If `RedirectResponse` is already imported at the top of the module, drop the local import line.)

- [ ] **Step 6: Delete settings.html**

After confirming all five sections' content was copied verbatim into the new templates:
```bash
git rm src/admin/templates/settings.html
```

- [ ] **Step 7: Run the section tests + full file**

Run: `python3 -m pytest tests/test_admin/test_settings_shell.py -v`
Expected: all pass (apply_settings tests + redirect + section render tests).

- [ ] **Step 8: Commit**

```bash
git add src/admin/routes.py src/admin/templates/settings_base.html src/admin/templates/settings_*.html tests/test_admin/test_settings_shell.py
git commit -m "feat: settings shell + per-section config pages (split from one form)"
```

---

## Task 3: Re-home the management pages into the shell

**Files:**
- Modify: `src/admin/templates/categories.html`, `proposals.html`, `connectors.html`, `hints.html`, `audit.html`
- Modify: `src/admin/routes.py` (the 5 management page routes add `"active"`)
- Test: `tests/test_admin/test_settings_shell.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_admin/test_settings_shell.py`:

```python
def test_management_pages_render_in_shell():
    from unittest.mock import patch, AsyncMock, MagicMock
    c = _client()
    # hints is the simplest (its route loads hints + datasets)
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store") as mg:
        store = AsyncMock()
        store.load_all_hints.return_value = []
        store.list_datasets.return_value = []
        mg.return_value = store
        resp = c.get("/admin/hints")
    assert resp.status_code == 200
    # rendered inside the settings shell -> the sub-nav (a link to another settings section) is present
    assert 'href="/admin/settings/security"' in resp.text
    # and Schema Hints is the active sub-nav item
    assert 'class="subnav-item active"' in resp.text
```

- [ ] **Step 2: Run to confirm fail**

Run: `python3 -m pytest tests/test_admin/test_settings_shell.py -k management_pages -v`
Expected: FAIL (hints.html still extends base.html → no sub-nav).

- [ ] **Step 3: Re-skin the 5 templates**

In each of `categories.html`, `proposals.html`, `connectors.html`, `hints.html`, `audit.html`:
- Change the first line `{% extends "base.html" %}` to `{% extends "settings_base.html" %}`.
- Change `{% block content %}` to `{% block settings_content %}` (the matching `{% endblock %}` stays).
- Leave ALL other markup, forms, htmx attributes, and scripts unchanged.

- [ ] **Step 4: Add `active` to the 5 management routes**

In `src/admin/routes.py`, find each route's `templates.TemplateResponse(request, "<page>.html", {<ctx>})` and add the `active` key to its context dict:
- `categories_page` → add `"active": "categories"`
- `proposals_page` → add `"active": "proposals"`
- `connectors_page` → add `"active": "connectors"`
- `hints_page` → add `"active": "hints"`
- `audit_page` → add `"active": "audit"`

Example (hints_page): change
```python
    return templates.TemplateResponse(request, "hints.html", {"groups": _build_hints_view(hints, datasets)})
```
to
```python
    return templates.TemplateResponse(request, "hints.html", {"groups": _build_hints_view(hints, datasets), "active": "hints"})
```
Do the equivalent one-key addition for the other four routes (keep each route's existing context keys).

- [ ] **Step 5: Run the test + full admin hint/settings files**

Run: `python3 -m pytest tests/test_admin/test_settings_shell.py tests/test_admin/test_hint_routes.py -v`
Expected: all pass (the hint-route tests still pass — hints.html now extends settings_base but still renders the groups + content).

- [ ] **Step 6: Commit**

```bash
git add src/admin/routes.py src/admin/templates/categories.html src/admin/templates/proposals.html src/admin/templates/connectors.html src/admin/templates/hints.html src/admin/templates/audit.html tests/test_admin/test_settings_shell.py
git commit -m "feat: re-home Categories/Proposals/Connectors/Schema Hints/Audit into the settings shell"
```

---

## Task 4: Top-nav cleanup

**Files:**
- Modify: `src/admin/templates/base.html`
- Test: `tests/test_admin/test_settings_shell.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_admin/test_settings_shell.py`:

```python
def test_top_nav_drops_moved_links():
    from unittest.mock import patch, AsyncMock, MagicMock
    c = _client()
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store") as mg:
        store = AsyncMock()
        store.list_documents.return_value = []
        store.list_categories.return_value = []
        store.list_proposals.return_value = []
        mg.return_value = store
        resp = c.get("/admin/")  # dashboard extends base.html, no settings sub-nav
    assert resp.status_code == 200
    nav = resp.text.split("<nav>")[1].split("</nav>")[0]
    for gone in ['/admin/categories', '/admin/proposals', '/admin/connectors', '/admin/hints', '/admin/audit']:
        assert gone not in nav, gone
    assert '/admin/settings' in nav  # Settings stays
```

- [ ] **Step 2: Run to confirm fail**

Run: `python3 -m pytest tests/test_admin/test_settings_shell.py -k top_nav -v`
Expected: FAIL (those links still in the top nav).

- [ ] **Step 3: Remove the 5 links from base.html**

In `src/admin/templates/base.html`, inside `<div class="nav-links">`, DELETE these five lines:
```html
            <a href="/admin/categories">Categories</a>
            <a href="/admin/proposals">Proposals</a>
            <a href="/admin/connectors">Connectors</a>
            <a href="/admin/hints">Schema Hints</a>
            <a href="/admin/audit">Audit Log</a>
```
Leave all other nav links (Dashboard, Datasets, Documents, Queue, Playground, Knowledge Graph, Settings, Logout, theme toggle) unchanged.

- [ ] **Step 4: Run the test**

Run: `python3 -m pytest tests/test_admin/test_settings_shell.py -k top_nav -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/admin/templates/base.html tests/test_admin/test_settings_shell.py
git commit -m "feat: shrink top nav (moved pages now live under Settings)"
```

---

## Task 5: Full suite + live smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the admin suite**

Run: `python3 -m pytest tests/test_admin/ -q`
Expected: the new test_settings_shell.py + test_hint_routes.py all pass; only the 5 PRE-EXISTING `test_routes.py` failures remain (unrelated auth-pattern rot — confirm no NEW failures).

- [ ] **Step 2: Live smoke (post-deploy)**

After rebuild + recreate of `sauron-api-1`, with an authenticated admin session in a browser:
- `/admin/settings` redirects to `/admin/settings/security`; the left sub-nav shows Configuration (Security/Models/Retrieval & Processing/System/Maintenance) and Management (Categories/Proposals/Connectors/Schema Hints/Audit Logs), with the active item highlighted.
- Open Models, change the LLM model, Save → "Settings saved successfully."; open Security and confirm its values are intact (partial save didn't clobber). Open Schema Hints / Audit Logs and confirm they render inside the shell with the sub-nav.
- Top nav no longer shows Categories/Proposals/Connectors/Schema Hints/Audit Log.

Quick container check that the routes resolve and the partial save round-trips:
```bash
docker exec sauron-api-1 python3 -c "
from src.admin.routes import _apply_settings_update
from src.config import settings
settings.admin_username='KEEP'; settings.llm_max_context=222222
p=_apply_settings_update({'vllm_model_name':'SMOKE'})
print('updated model:', settings.vllm_model_name, '| kept admin:', settings.admin_username, '| kept ctx:', settings.llm_max_context)
print('persist keeps both:', p['admin_username']=='KEEP' and p['llm_max_context']==222222)
"
```
Expected: `updated model: SMOKE | kept admin: KEEP | kept ctx: 222222` and `persist keeps both: True`.

- [ ] **Step 3: Final status**

```bash
git status   # clean
```

---

## Self-Review Notes (completed by plan author)

- **Spec coverage:** settings shell + left sub-nav (Task 2 `settings_base.html`, spec §1); top-nav cleanup (Task 4, spec §2); config split + per-section save + partial merge (Tasks 1 & 2, spec §3, the risk covered by Task 1 tests incl. the checkbox hidden-field pattern from spec §Risks); management re-home keeping routes + `active` context (Task 3, spec §4); section list incl. Retrieval kept (Task 2 templates + shell). Tests for shell/section/partial-save/re-home/top-nav (spec §Testing). No gaps.
- **Type/signature consistency:** `_apply_settings_update(form) -> dict` defined in Task 1, used by `save_settings` (Task 1) and the live smoke (Task 5). `active` context key produced by config routes (Task 2) and management routes (Task 3), consumed by `settings_base.html` (Task 2). The `subnav-item active` class string asserted in tests matches the class emitted by `settings_base.html`.
- **Placeholder scan:** none — full code for shell/handler/routes/nav; the verbatim-copy instructions reference exact `settings.html` line ranges (the implementer copies existing markup rather than the plan reproducing 300 lines of HTML), with the checkbox hidden-field pattern given explicitly.
