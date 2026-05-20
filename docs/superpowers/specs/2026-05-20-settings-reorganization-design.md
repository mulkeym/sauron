# Settings Page Reorganization

## Problem

The settings page mixes configuration and maintenance actions, has inconsistent styling (some sections use `settings-section` wrappers, others are bare `<h3>` tags), splits related settings across distant sections (entity reconciliation thresholds vs run button), and has confusing save scope (two separate forms for security, one big form for everything else).

## Design

Single scrolling page. One form wrapping all configuration. One Save button. Maintenance actions separated visually below.

### Layout

```
┌─────────────────────────────────────────┐
│ Settings                                │
├─────────────────────────────────────────┤
│ SECURITY                                │
│  Admin Account: username, password      │
│  API Keys: input + Generate Key button  │
├─────────────────────────────────────────┤
│ MODELS                                  │
│  LLM Inference:                         │
│    Base URL + Refresh Models button     │
│    API Key                              │
│    Model dropdown                       │
│    Test Connection button               │
│  Embedding Model:                       │
│    Mode (Local/API)                     │
│    [if API] Base URL + Refresh Models   │
│    [if API] Model dropdown              │
│    [if API] Test Connection             │
├─────────────────────────────────────────┤
│ RETRIEVAL & PROCESSING                  │
│  LLM Limits:                            │
│    Max Context (chars) | Max Output Tkns│
│  Concurrency:                           │
│    Parallel Ingestion | LLM Concurrency │
│  Metadata Extraction:                   │
│    Enable/Disable | Max Doc Length      │
│  Retrieval Tuning:                      │
│    Feedback Enable | Similarity Thresh  │
│    Query Expansion (PRF) Enable         │
│    Strategy Memory Enable               │
│    Entity Recon: Auto Thresh | Review   │
├─────────────────────────────────────────┤
│ SYSTEM                                  │
│  MCP Server: Port                       │
│  Vector Database: Path, Table (readonly)│
├─────────────────────────────────────────┤
│          [ Save Settings ]              │
├═════════════════════════════════════════╡
│ MAINTENANCE                             │
│  (visually distinct - different bg)     │
│                                         │
│  Query Result Cache:                    │
│    Purge button + entry count           │
│                                         │
│  Knowledge Graph:                       │
│    Purge button + Backfill Metadata btn │
│                                         │
│  Entity Reconciliation:                 │
│    Run Now + Stop buttons + status      │
│                                         │
│  Query Performance Metrics:             │
│    Dashboard + Reset Feedback Data btn  │
│                                         │
│  Backup & Restore:                      │
│    Create Backup + backup list          │
│    Restore upload form                  │
└─────────────────────────────────────────┘
```

### Changes from Current State

**Structural:**
- Merge Admin Account and API Keys forms into the single global form. Remove their separate save buttons.
- Wrap Concurrency, Metadata Extraction, and Relevance Feedback in proper `settings-section` divs (currently bare `<h3>` tags inside the form).
- Move Entity Reconciliation thresholds into Retrieval & Processing section. Keep the Run button in Maintenance.
- Add a visual divider (styled `<hr>` or distinct background) between Save button and Maintenance area.

**Reordering:**
1. Security (Admin Account, API Keys)
2. Models (LLM Inference, Embedding Model)
3. Retrieval & Processing (LLM Limits, Concurrency, Metadata Extraction, Retrieval Tuning with feedback/PRF/strategy memory/entity recon thresholds)
4. System (MCP port, LanceDB read-only info)
5. Save Settings button
6. Maintenance (Cache, Knowledge Graph, Entity Reconciliation run, Metrics, Backup & Restore)

**Backend:**
- Update `save_settings` endpoint to also accept `admin_username`, `admin_password`, and `api_keys` fields. Remove the separate `/api/settings/admin-account` and `/api/settings/api-keys` POST handlers.
- Persist `admin_username`, `admin_password`, and `api_keys` to `data/settings.json` alongside existing fields.

**Styling:**
- Maintenance area gets a distinct visual treatment: muted background color, slightly different border, and a "Maintenance" heading to clearly separate it from configuration.
- All config sections use consistent `settings-section` class wrappers.
- Sub-groups within a section (e.g., "LLM Limits", "Concurrency") use `<h3>` sub-headers within the parent `settings-section`.

### Files to Modify

- `src/admin/templates/settings.html` — reorganize template structure
- `src/admin/routes.py` — merge save endpoints, update `save_settings` to handle security fields
- `src/config.py` — add `admin_username`, `admin_password`, `api_keys` to persisted settings
- `src/admin/static/style.css` — add maintenance area styling (if needed)

### What Stays the Same

- All existing fields and their behavior (conditional embedding fields, test connection buttons, refresh model buttons)
- HTMX interactions for test buttons, metrics loading, backup status polling
- The actual settings values and their defaults
- No new configuration fields added
