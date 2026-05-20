# Settings Page Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the settings page into logical groups (Security, Models, Retrieval & Processing, System) with a single Save button for all config, and a visually distinct Maintenance area for action buttons.

**Architecture:** Single-page layout, one `<form>` wrapping all config sections. Backend `save_settings` endpoint absorbs admin account and API key fields. Separate admin-account and api-keys endpoints removed. Maintenance section outside the form, styled distinctly.

**Tech Stack:** Jinja2 templates, HTMX, FastAPI, CSS

---

### Task 1: Add maintenance area CSS

**Files:**
- Modify: `src/admin/static/style.css`

- [ ] **Step 1: Add maintenance section styles**

Add after the existing `.form-row .form-group` rule (line 90):

```css
.maintenance-area { background: var(--bg-card); border: 2px dashed var(--border); border-radius: 8px; padding: 1.5rem; margin-top: 2rem; }
.maintenance-area h2 { font-size: 1.2rem; margin-bottom: 1rem; color: var(--text-muted); }
.maintenance-area .settings-section { box-shadow: none; padding: 1rem; margin-bottom: 1rem; }
```

- [ ] **Step 2: Commit**

```bash
git add src/admin/static/style.css
git commit -m "style: add maintenance area CSS for settings page"
```

---

### Task 2: Merge security endpoints into save_settings

**Files:**
- Modify: `src/admin/routes.py:1693-1737` (remove `update_admin_account` and `update_api_keys`)
- Modify: `src/admin/routes.py:1741-1812` (update `save_settings` signature and persist logic)

- [ ] **Step 1: Add admin_username, admin_password, api_keys params to save_settings**

In `src/admin/routes.py`, update the `save_settings` function signature to add these three params at the top:

```python
@router.post("/api/settings")
async def save_settings(
    admin_username: str = Form(""),
    admin_password: str = Form(""),
    api_keys: str = Form(""),
    vllm_base_url: str = Form(""),
    vllm_model_name: str = Form(""),
    vllm_api_key: str = Form(""),
    embedding_mode: str = Form(""),
    embedding_api_url: str = Form(""),
    embedding_model_name: str = Form(""),
    mcp_port: int = Form(8090),
    entity_merge_auto_threshold: float = Form(0.9),
    entity_merge_review_threshold: float = Form(0.7),
    max_parallel_ingestion: int = Form(3),
    llm_concurrency: int = Form(4),
    llm_max_context: int = Form(200000),
    llm_max_output_tokens: int = Form(32768),
    metadata_extraction_enabled: bool = Form(True),
    metadata_max_doc_length: int = Form(200000),
    feedback_enabled: bool = Form(True),
    feedback_similarity_threshold: float = Form(0.85),
    prf_enabled: bool = Form(True),
    strategy_memory_enabled: bool = Form(True),
):
```

- [ ] **Step 2: Add security fields to the in-memory update block**

Add at the top of the function body, before the existing `if vllm_base_url:` block:

```python
    # Security
    if admin_username:
        settings.admin_username = admin_username
    if admin_password:
        settings.admin_password = admin_password
    if api_keys.strip():
        settings.api_keys = api_keys.strip()
```

- [ ] **Step 3: Add security fields to the persist dict**

Add to the `persist = {` dict:

```python
        "admin_username": settings.admin_username,
        "admin_password": settings.admin_password,
        "api_keys": settings.api_keys,
```

- [ ] **Step 4: Remove the old separate endpoints**

Delete the `update_admin_account` function (the `@router.post("/api/settings/admin-account")` handler and its full body) and the `update_api_keys` function (the `@router.post("/api/settings/api-keys")` handler and its full body). These are at lines ~1693-1737.

- [ ] **Step 5: Commit**

```bash
git add src/admin/routes.py
git commit -m "refactor: merge admin account and API key endpoints into save_settings"
```

---

### Task 3: Reorganize the settings template

**Files:**
- Rewrite: `src/admin/templates/settings.html`

This is the main task. Replace the entire template content with the new layout. The file structure:

```
{% extends "base.html" %}
{% block title %}Settings - SAURON{% endblock %}
{% block content %}
<h1>Settings</h1>

<form hx-post="/admin/api/settings" hx-target="#save-status" hx-swap="innerHTML">

  <!-- Section 1: Security -->
  <!-- Section 2: Models -->
  <!-- Section 3: Retrieval & Processing -->
  <!-- Section 4: System -->

  <div class="form-actions">
    <button type="submit">Save Settings</button>
    <span id="save-status" style="margin-left:0.5rem;"></span>
  </div>
</form>

<!-- Maintenance area (outside form) -->
<div class="maintenance-area">
  <h2>Maintenance</h2>
  <!-- Cache, KG, Reconciliation, Metrics, Backup -->
</div>

{% endblock %}
```

- [ ] **Step 1: Write Section 1 — Security**

Replace the current separate Admin Account and API Keys forms with fields inside the single form:

```html
<div class="settings-section">
    <h2>Security</h2>
    <p class="section-desc">Admin login credentials and API access keys.</p>
    <div class="form-row">
        <div class="form-group">
            <label for="admin_username">Admin Username</label>
            <input type="text" id="admin_username" name="admin_username" value="{{ settings.admin_username }}">
        </div>
        <div class="form-group">
            <label for="admin_password">Admin Password</label>
            <input type="password" id="admin_password" name="admin_password" placeholder="Leave blank to keep current">
        </div>
    </div>
    <div class="form-group">
        <label for="api_keys">API Keys</label>
        <div class="input-with-button">
            <input type="text" id="api_keys" name="api_keys" value="{{ settings.api_keys }}" style="font-family:monospace;">
            <button type="button" class="secondary" onclick="generateApiKey()">Generate Key</button>
        </div>
        <span style="font-size:0.8rem; color:#6b7280;">Comma-separated. Each key can be used as X-API-Key header or Bearer token.</span>
    </div>
</div>

<script>
function generateApiKey() {
    const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
    let key = 'sk-';
    for (let i = 0; i < 32; i++) key += chars[Math.floor(Math.random() * chars.length)];
    const input = document.getElementById('api_keys');
    input.value = input.value ? input.value + ',' + key : key;
}
</script>
```

- [ ] **Step 2: Write Section 2 — Models**

LLM Inference and Embedding Model, identical fields to current but grouped:

```html
<div class="settings-section">
    <h2>Models</h2>
    <p class="section-desc">LLM and embedding model configuration.</p>

    <h3>LLM Inference</h3>
    <div class="form-group">
        <label for="vllm_base_url">API Base URL</label>
        <div class="input-with-button">
            <input type="text" id="vllm_base_url" name="vllm_base_url" value="{{ settings.vllm_base_url }}" placeholder="http://gpu-server:8000/v1">
            <button type="button" class="secondary" hx-post="/admin/api/settings/list-llm-models" hx-include="#vllm_base_url,#vllm_api_key" hx-target="#llm-model-select" hx-swap="innerHTML">Refresh Models</button>
        </div>
    </div>
    <div class="form-group">
        <label for="vllm_api_key">API Key</label>
        <input type="password" id="vllm_api_key" name="vllm_api_key" value="{{ settings.vllm_api_key }}" placeholder="sk-... (leave empty for local models)">
        <span style="font-size:0.8rem; color:#6b7280;">Required for OpenAI, Anthropic, etc. Leave empty for local endpoints.</span>
    </div>
    <div class="form-group">
        <label for="vllm_model_name">Model</label>
        <div id="llm-model-select">
            <select name="vllm_model_name" id="vllm_model_name">
                <option value="{{ settings.vllm_model_name }}" selected>{{ settings.vllm_model_name }}</option>
            </select>
        </div>
    </div>
    <div class="form-group">
        <button type="button" class="secondary" hx-post="/admin/api/settings/test-llm" hx-include="#vllm_base_url,#vllm_api_key" hx-target="#llm-status" hx-swap="innerHTML">Test Connection</button>
        <span id="llm-status"></span>
    </div>

    <h3 style="margin-top:1.5rem;">Embedding Model</h3>
    <div class="form-group">
        <label for="embedding_mode">Mode</label>
        <select id="embedding_mode" name="embedding_mode">
            <option value="api" {% if settings.embedding_mode == "api" %}selected{% endif %}>API (external endpoint)</option>
            <option value="local" {% if settings.embedding_mode == "local" %}selected{% endif %}>Local (sentence-transformers)</option>
        </select>
    </div>
    <div id="embedding-api-fields" style="{% if settings.embedding_mode == 'local' %}display:none;{% endif %}">
        <div class="form-group">
            <label for="embedding_api_url">API Base URL</label>
            <div class="input-with-button">
                <input type="text" id="embedding_api_url" name="embedding_api_url" value="{{ settings.embedding_api_url }}" placeholder="http://gpu-server:8000/v1">
                <button type="button" class="secondary" hx-post="/admin/api/settings/list-embedding-models" hx-include="#embedding_api_url" hx-target="#embed-model-select" hx-swap="innerHTML">Refresh Models</button>
            </div>
        </div>
        <div class="form-group">
            <label for="embedding_model_name">Model</label>
            <div id="embed-model-select">
                <select name="embedding_model_name" id="embedding_model_name">
                    <option value="{{ settings.embedding_model_name }}" selected>{{ settings.embedding_model_name }}</option>
                </select>
            </div>
        </div>
        <div class="form-group">
            <button type="button" class="secondary" hx-post="/admin/api/settings/test-embedding" hx-include="#embedding_mode, #embedding_api_url, #embedding_model_name" hx-target="#embed-status" hx-swap="innerHTML">Test Connection</button>
            <span id="embed-status"></span>
        </div>
    </div>
    <script>
        document.getElementById('embedding_mode').addEventListener('change', function() {
            document.getElementById('embedding-api-fields').style.display = this.value === 'local' ? 'none' : '';
        });
    </script>
</div>
```

- [ ] **Step 3: Write Section 3 — Retrieval & Processing**

Consolidates concurrency, LLM limits, metadata extraction, and retrieval tuning:

```html
<div class="settings-section">
    <h2>Retrieval & Processing</h2>
    <p class="section-desc">Tuning for document retrieval, LLM context, and learning systems.</p>

    <h3>LLM Limits</h3>
    <div class="form-row">
        <div class="form-group">
            <label for="llm_max_context">Max Context (chars)</label>
            <input type="number" id="llm_max_context" name="llm_max_context" value="{{ settings.llm_max_context }}" min="10000" max="2000000" step="10000" style="max-width:140px;">
            <span style="font-size:0.8rem; color:#6b7280;">200K ~ 50K tokens. Reduce for smaller models.</span>
        </div>
        <div class="form-group">
            <label for="llm_max_output_tokens">Max Output Tokens</label>
            <input type="number" id="llm_max_output_tokens" name="llm_max_output_tokens" value="{{ settings.llm_max_output_tokens }}" min="1024" max="131072" step="1024" style="max-width:140px;">
            <span style="font-size:0.8rem; color:#6b7280;">Max tokens per LLM response. 32K for exhaustive listings.</span>
        </div>
    </div>

    <h3>Concurrency</h3>
    <div class="form-row">
        <div class="form-group">
            <label for="max_parallel_ingestion">Parallel Ingestion Jobs</label>
            <input type="number" id="max_parallel_ingestion" name="max_parallel_ingestion" value="{{ settings.max_parallel_ingestion }}" min="1" max="20" style="max-width:100px;">
            <span style="font-size:0.8rem; color:#6b7280;">Concurrent file processing workers</span>
        </div>
        <div class="form-group">
            <label for="llm_concurrency">LLM Concurrency</label>
            <input type="number" id="llm_concurrency" name="llm_concurrency" value="{{ settings.llm_concurrency }}" min="1" max="32" style="max-width:100px;">
            <span style="font-size:0.8rem; color:#6b7280;">Concurrent LLM calls (map-reduce, extraction)</span>
        </div>
    </div>

    <h3>Metadata Extraction</h3>
    <div class="form-row">
        <div class="form-group">
            <label for="metadata_extraction_enabled">Enable Metadata Extraction</label>
            <select id="metadata_extraction_enabled" name="metadata_extraction_enabled">
                <option value="true" {{ 'selected' if settings.metadata_extraction_enabled }}>Enabled</option>
                <option value="false" {{ 'selected' if not settings.metadata_extraction_enabled }}>Disabled</option>
            </select>
        </div>
        <div class="form-group">
            <label for="metadata_max_doc_length">Max Document Length (chars)</label>
            <input type="number" id="metadata_max_doc_length" name="metadata_max_doc_length" value="{{ settings.metadata_max_doc_length }}" min="1000" max="500000" style="max-width:120px;">
            <span style="font-size:0.8rem; color:#6b7280;">Chars sent to LLM for extraction</span>
        </div>
    </div>

    <h3>Retrieval Tuning</h3>
    <div class="form-row">
        <div class="form-group">
            <label for="feedback_enabled">Relevance Feedback</label>
            <select id="feedback_enabled" name="feedback_enabled">
                <option value="true" {{ 'selected' if settings.feedback_enabled }}>Enabled</option>
                <option value="false" {{ 'selected' if not settings.feedback_enabled }}>Disabled</option>
            </select>
            <span style="font-size:0.8rem; color:#6b7280;">Learn from query results to improve retrieval</span>
        </div>
        <div class="form-group">
            <label for="feedback_similarity_threshold">Similarity Threshold</label>
            <input type="number" id="feedback_similarity_threshold" name="feedback_similarity_threshold" value="{{ settings.feedback_similarity_threshold }}" step="0.05" min="0.5" max="1.0" style="max-width:100px;">
            <span style="font-size:0.8rem; color:#6b7280;">Min cosine similarity to match past queries</span>
        </div>
    </div>
    <div class="form-row">
        <div class="form-group">
            <label for="prf_enabled">Query Expansion (PRF)</label>
            <select id="prf_enabled" name="prf_enabled">
                <option value="true" {{ 'selected' if settings.prf_enabled }}>Enabled</option>
                <option value="false" {{ 'selected' if not settings.prf_enabled }}>Disabled</option>
            </select>
            <span style="font-size:0.8rem; color:#6b7280;">Expand queries with terms from top results</span>
        </div>
        <div class="form-group">
            <label for="strategy_memory_enabled">Strategy Memory</label>
            <select id="strategy_memory_enabled" name="strategy_memory_enabled">
                <option value="true" {{ 'selected' if settings.strategy_memory_enabled }}>Enabled</option>
                <option value="false" {{ 'selected' if not settings.strategy_memory_enabled }}>Disabled</option>
            </select>
            <span style="font-size:0.8rem; color:#6b7280;">Learn which strategy works best per query pattern</span>
        </div>
    </div>
    <div class="form-row">
        <div class="form-group">
            <label for="entity_merge_auto_threshold">Entity Auto-merge Threshold</label>
            <input type="number" id="entity_merge_auto_threshold" name="entity_merge_auto_threshold" value="{{ settings.entity_merge_auto_threshold }}" step="0.05" min="0" max="1" style="max-width:100px;">
            <span style="font-size:0.8rem; color:#6b7280;">Above this confidence, merge automatically</span>
        </div>
        <div class="form-group">
            <label for="entity_merge_review_threshold">Entity Review Threshold</label>
            <input type="number" id="entity_merge_review_threshold" name="entity_merge_review_threshold" value="{{ settings.entity_merge_review_threshold }}" step="0.05" min="0" max="1" style="max-width:100px;">
            <span style="font-size:0.8rem; color:#6b7280;">Above this, propose for admin review</span>
        </div>
    </div>
</div>
```

- [ ] **Step 4: Write Section 4 — System**

```html
<div class="settings-section">
    <h2>System</h2>
    <div class="form-group">
        <label for="mcp_port">MCP Server Port</label>
        <input type="number" id="mcp_port" name="mcp_port" value="{{ settings.mcp_port }}" placeholder="8090" style="max-width:100px;">
    </div>
    <div class="form-row">
        <div class="form-group">
            <label>Vector Database Path</label>
            <input type="text" value="{{ settings.lancedb_path }}" disabled>
        </div>
        <div class="form-group">
            <label>Vector Database Table</label>
            <input type="text" value="{{ settings.lancedb_table_name }}" disabled>
        </div>
    </div>
    <span style="font-size:0.8rem; color:#6b7280;">LanceDB is embedded — no server required. Data stored locally.</span>
</div>
```

- [ ] **Step 5: Write the Save button and Maintenance area**

Save button between config and maintenance:

```html
<div class="form-actions">
    <button type="submit">Save Settings</button>
    <span id="save-status" style="margin-left:0.5rem;"></span>
</div>
</form>

<div class="maintenance-area">
    <h2>Maintenance</h2>

    <div class="settings-section">
        <h3>Query Result Cache</h3>
        <p class="section-desc">Cached answers are reused for semantically similar questions. Purge to force fresh answers.</p>
        <button type="button" hx-post="/admin/api/settings/purge-cache" hx-target="#cache-status" hx-swap="innerHTML" hx-confirm="Purge all cached query results?">Purge Cache</button>
        <span id="cache-status" style="margin-left:0.5rem;"
              hx-get="/admin/api/settings/cache-stats" hx-trigger="load" hx-swap="innerHTML">
        </span>
    </div>

    <div class="settings-section">
        <h3>Knowledge Graph</h3>
        <p class="section-desc">Purge the LightRAG knowledge graph or backfill metadata for older documents.</p>
        <button type="button" hx-post="/admin/api/settings/purge-knowledge-graph" hx-target="#kg-status" hx-swap="innerHTML" hx-confirm="This will delete ALL knowledge graph data. Continue?">Purge Knowledge Graph</button>
        <span id="kg-status" style="margin-left:0.5rem;"></span>
        <div style="margin-top:0.75rem;">
            <button type="button" hx-post="/admin/api/settings/backfill-metadata" hx-target="#metadata-status" hx-swap="innerHTML">Backfill Document Metadata</button>
            <span id="metadata-status" style="margin-left:0.5rem;"></span>
            <p class="section-desc" style="margin-top:0.25rem;">Extract metadata for documents ingested before this feature was added.</p>
        </div>
    </div>

    <div class="settings-section">
        <h3>Entity Reconciliation</h3>
        <p class="section-desc">Scan entities for duplicates. High-confidence matches merge automatically, others go to Knowledge Graph for review.</p>
        <button type="button" hx-post="/admin/api/settings/reconcile" hx-target="#reconcile-status" hx-swap="innerHTML">Run Now</button>
        <button type="button" hx-post="/admin/api/settings/reconcile-stop" hx-target="#reconcile-status" hx-swap="innerHTML" style="margin-left:0.5rem;">Stop</button>
        <div id="reconcile-status" style="margin-top:0.5rem;"
             hx-get="/admin/api/settings/reconcile-status" hx-trigger="load" hx-swap="innerHTML">
        </div>
    </div>

    <div class="settings-section">
        <h3>Query Performance Metrics</h3>
        <p class="section-desc">Retrieval accuracy and efficiency. MAP Precision = % of documents the LLM read that were relevant.</p>
        <div id="metrics-dashboard" hx-get="/admin/api/settings/query-metrics" hx-trigger="load" hx-swap="innerHTML">
        </div>
        <div style="margin-top:0.5rem;">
            <button type="button" hx-post="/admin/api/settings/purge-feedback" hx-target="#purge-fb-status" hx-swap="innerHTML" hx-confirm="Clear all feedback and metrics data?">Reset Feedback Data</button>
            <span id="purge-fb-status" style="margin-left:0.5rem;"></span>
        </div>
    </div>

    <div class="settings-section">
        <h3>Backup & Restore</h3>
        <p class="section-desc">Create a backup of all data and configuration. Download to transport to another host.</p>
        <div style="margin-bottom:1rem;">
            <button type="button" hx-post="/admin/api/backup/create" hx-target="#backup-status" hx-swap="innerHTML">Create Backup</button>
            <span id="backup-status" style="margin-left:0.5rem;"
                  hx-get="/admin/api/backup/status" hx-trigger="every 1s" hx-swap="innerHTML">
            </span>
        </div>
        <div id="backup-list" hx-get="/admin/api/backup/list" hx-trigger="load, every 5s" hx-swap="innerHTML" style="margin-bottom:1rem;">
        </div>
        <h4 style="margin-bottom:0.5rem;">Restore from Backup</h4>
        <p class="section-desc">Upload a backup file to restore. Current data is saved before overwriting. Server restart required after restore.</p>
        <form hx-post="/admin/api/backup/restore" hx-target="#restore-status" hx-swap="innerHTML" hx-encoding="multipart/form-data">
            <div class="form-group">
                <input type="file" name="backup_file" accept=".gz,.tar.gz" required>
            </div>
            <button type="submit" hx-confirm="This will overwrite all current data. Continue?">Restore</button>
            <span id="restore-status" style="margin-left:0.5rem;"></span>
        </form>
    </div>
</div>
```

- [ ] **Step 6: Assemble the full template**

Combine all sections into the complete `settings.html` file. The full structure:

```
{% extends "base.html" %}
{% block title %}Settings - SAURON{% endblock %}
{% block content %}
<h1>Settings</h1>
<form hx-post="/admin/api/settings" hx-target="#save-status" hx-swap="innerHTML">
  [Section 1: Security from Step 1]
  [Section 2: Models from Step 2]
  [Section 3: Retrieval & Processing from Step 3]
  [Section 4: System from Step 4]
  [Save button]
</form>
[Maintenance area from Step 5]
{% endblock %}
```

- [ ] **Step 7: Commit**

```bash
git add src/admin/templates/settings.html
git commit -m "refactor: reorganize settings page layout into logical groups"
```

---

### Task 4: Verify and push

- [ ] **Step 1: Rebuild and test locally**

```bash
docker compose up --build -d
```

Open http://localhost:8080/admin/settings and verify:
- All 4 config sections render correctly with proper grouping
- Save Settings button saves all fields including admin username, password, API keys
- Embedding API fields hide/show when mode changes
- Test Connection and Refresh Models buttons still work
- Maintenance area is visually distinct below the save button
- All maintenance actions (purge cache, purge KG, run reconciliation, backup) work
- Settings persist after `docker compose down && docker compose up -d`

- [ ] **Step 2: Push**

```bash
git push origin master
```
