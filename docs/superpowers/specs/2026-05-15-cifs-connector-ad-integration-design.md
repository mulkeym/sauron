# CIFS Connector with Active Directory Integration

## Overview

Add a CIFS/SMB file share connector that scans Windows file shares, reads NTFS ACLs, maps them to SAURON ACL groups via Active Directory group resolution, and ingests files with proper access control. Users authenticate via AD/LDAP, with group memberships resolved at login time.

This is a 3-phase feature. Each phase is independently useful and builds on the previous.

## Phase 1: ACL Group Management UI

### Goal

Admin page to manage SAURON ACL groups and map them to Active Directory Distinguished Names. This is the foundation for both CIFS ACL resolution and LDAP-based login.

### Changes

**New admin page: `/admin/acl-groups`**
- List all ACL groups with columns: Name, Display Name, AD Group DN, Description, Active, Actions
- Create form: name, display_name, ad_group_dn, description
- Inline edit (same pattern as web connectors)
- Activate/deactivate toggle

**New admin routes:**
- `GET /admin/acl-groups` — page
- `POST /admin/api/acl-groups/create` — create group
- `POST /admin/api/acl-groups/{id}/update` — edit group
- `POST /admin/api/acl-groups/{id}/toggle` — activate/deactivate

**Existing infrastructure (no changes needed):**
- `AclGroup` model already has: `id`, `name`, `display_name`, `ad_group_dn`, `description`, `active`, `created_at`
- `MetadataStore` already has: `add_acl_group()`, `list_acl_groups()`, `get_acl_group_names()`
- Navigation sidebar needs a link to "ACL Groups"

---

## Phase 2: AD/LDAP Authentication

### Goal

Users log in with Active Directory credentials. Their AD group memberships are resolved to SAURON ACL group names via the AclGroup mapping table, then embedded in the JWT token.

### New Module: `src/auth/ldap.py`

**Functions:**
- `ldap_authenticate(username, password) -> dict | None` — bind to AD, return user info + group memberships
- `resolve_ad_groups_to_sauron(ad_group_dns: list[str]) -> list[str]` — map AD group DNs to SAURON ACL group names via `AclGroup.ad_group_dn`

**LDAP flow:**
1. User enters username + password on login page
2. System attempts LDAP bind: `LDAP_BIND_DN` template with username (e.g. `{username}@corp.com`)
3. On success, query `memberOf` attribute to get AD group DNs
4. Map each AD group DN to a SAURON ACL group name via `AclGroup.ad_group_dn`
5. Create JWT with `groups: ["finance", "executives", ...]`
6. Fallback: if LDAP is disabled or unreachable, fall back to local admin auth

### Configuration (src/config.py)

```
LDAP_ENABLED=true
LDAP_SERVER=ldap://dc01.corp.com
LDAP_BASE_DN=DC=corp,DC=com
LDAP_BIND_TEMPLATE={username}@corp.com
LDAP_GROUP_SEARCH_BASE=OU=Groups,DC=corp,DC=com
LDAP_USE_SSL=true
LDAP_TIMEOUT=10
```

### Login Page Changes

- When `LDAP_ENABLED=true`, the login form sends credentials to LDAP first
- If LDAP auth fails, try local admin auth as fallback
- Show "Logged in as {username} via Active Directory" in the UI

### Docker Additions

- `libldap2-dev`, `libsasl2-dev` in Dockerfile for `python-ldap` package
- Or use `ldap3` (pure Python, no system deps)

---

## Phase 3: CIFS Connector

### Goal

Scan a Windows file share, read NTFS file ACLs, resolve AD groups to SAURON ACL groups, and ingest files. Full UNC path stored as `source_url` for citations.

### New Model: `CIFSConnector`

```
__tablename__ = "cifs_connectors"

id: int (PK, autoincrement)
name: str (required)
smb_path: str (required) — UNC path, e.g. \\server\share\folder
dataset_id: int — which dataset to assign documents to
file_patterns: list[str] (JSON) — glob include patterns, e.g. ["*.pdf", "*.docx", "*.pptx", "*.xlsx"]
exclude_patterns: list[str] (JSON) — path exclude patterns, e.g. ["*/Archive/*", "*/Temp/*"]
scan_depth: int — 0=root only, 1-N=levels deep, -1=unlimited
acl_mode: str — "ntfs" (read file ACLs) or "static" (use acl_groups field)
acl_groups: list[str] (JSON) — static ACL groups (used when acl_mode="static")
category: str — document category (blank = auto-categorize)
active: bool
last_scan: datetime (nullable)
files_found: int
files_ingested: int
created_at: datetime
```

### New Module: `src/ingestion/cifs_scanner.py`

**SMB library:** `smbprotocol` (pure Python, supports SMBv2/v3, Kerberos and NTLM auth)

**Authentication:**
- Primary: Kerberos via system keytab (container is domain-joined)
- Fallback: NTLM with username/password from environment (`CIFS_USERNAME`, `CIFS_PASSWORD`, `CIFS_DOMAIN`)
- Auth method selected per environment, not per connector (all connectors use the same service account)

**Scan flow:**

```
1. Connect to SMB share using Kerberos or NTLM credentials
2. Recursively list directories (respecting scan_depth)
3. For each file:
   a. Check file_patterns (include) and exclude_patterns (exclude)
   b. Check last_modified timestamp against previous scan
   c. If new/changed: read file content, compute SHA-256 hash
   d. Check content_hash against metadata DB for dedup
   e. If acl_mode="ntfs":
      - Read file security descriptor via SMB
      - Extract SIDs with FILE_READ_DATA permission
      - Resolve SIDs to AD group DNs (LDAP lookup, cached)
      - Map AD group DNs to SAURON ACL group names via AclGroup table
      - Auto-create AclGroup entries for unmapped AD groups
   f. If acl_mode="static":
      - Use connector's acl_groups field
   g. Download file to temp directory
   h. Enqueue via ingest_queue.enqueue():
      - filename: original filename from share
      - source_url: smb://server/share/path/file.pdf
      - acl_groups: resolved groups
      - dataset_id: from connector
      - category: from connector (or auto-categorize)
      - uploaded_by: "cifs-connector"
4. Update connector stats (files_found, files_ingested, last_scan)
```

**SID → AD Group DN resolution:**
- On first encounter, query LDAP: `(&(objectClass=group)(objectSID=<sid>))` → get `distinguishedName`
- Cache SID → DN mapping in memory for the duration of the scan
- Map DN to SAURON group name via `AclGroup.ad_group_dn`
- If no AclGroup has that DN, auto-create one:
  - `name`: sanitized from AD group `cn` (e.g. "CN=Finance Team" → "finance_team")
  - `ad_group_dn`: the full DN
  - `display_name`: the AD group `cn`
  - `active`: True

**Incremental sync:**
- Track `{file_path: {last_modified, content_hash}}` in a JSON file per connector: `data/cifs_scan_state_{connector_id}.json`
- On scan: skip files where `last_modified` hasn't changed
- For changed files: recompute hash, re-ingest if hash differs
- Detect deleted files: files in previous state but not in current scan (log only, don't delete from SAURON)

### Admin UI

**New page: `/admin/cifs-connectors`** (same pattern as web connectors)

- Create form: name, smb_path, dataset, file_patterns, exclude_patterns, scan_depth, acl_mode, category, acl_groups (for static mode)
- Connector list table: Name, SMB Path, Files Found/Ingested, Last Scan, Status, Actions
- Inline edit form
- "Scan Now" button with live progress on Queue page
- Active crawl status tracker (same pattern as web connector crawl status)

**New routes:**
- `GET /admin/cifs-connectors` — page
- `POST /admin/api/cifs-connectors/create`
- `POST /admin/api/cifs-connectors/{id}/update`
- `POST /admin/api/cifs-connectors/{id}/scan` — trigger async scan
- `DELETE /admin/api/cifs-connectors/{id}` — deactivate

### Docker Additions for Domain Join

```dockerfile
RUN apt-get install -y --no-install-recommends \
    krb5-user libkrb5-3 libgssapi-krb5-2
```

**Volume mounts in docker-compose.yml:**
- `/etc/krb5.conf` — Kerberos realm configuration
- `/etc/krb5.keytab` — Service account keytab for domain auth

**Container init:** `kinit -kt /etc/krb5.keytab sauron_svc@CORP.COM` on startup (or use `smbprotocol`'s built-in Kerberos support)

### Configuration (src/config.py)

```
CIFS_AUTH_METHOD=kerberos  # kerberos or ntlm
CIFS_DOMAIN=CORP.COM
CIFS_USERNAME=  # only for NTLM fallback
CIFS_PASSWORD=  # only for NTLM fallback
CIFS_KEYTAB_PATH=/etc/krb5.keytab
CIFS_SERVICE_PRINCIPAL=sauron_svc@CORP.COM
```

---

## Data Flow Summary

```
Phase 1: ACL Group Registry
  Admin creates/maps ACL groups ↔ AD group DNs
  
Phase 2: User Login via AD
  User login → LDAP bind → resolve memberOf → map AD DNs to SAURON groups → JWT

Phase 3: CIFS File Ingestion
  Connector scans share → reads file NTFS ACLs → resolves SIDs to AD DNs
  → maps to SAURON groups (auto-creates unmapped) → downloads file
  → enqueue with acl_groups + source_url → standard ingestion pipeline
  → vector store with ACL tags → search respects user's JWT groups
```

## Error Handling

- **SMB connection failure**: log error, mark connector scan as failed, retry on next scheduled scan
- **Kerberos ticket expired**: attempt `kinit` refresh before scan; if fails, fall back to NTLM if configured
- **LDAP unreachable during SID resolution**: use cached mappings if available; skip ACL resolution for new SIDs (log warning)
- **File read permission denied**: skip file, log warning, continue scan
- **Unmapped AD group**: auto-create AclGroup entry, log info

## Files Created/Modified

### Phase 1
- Create: `src/admin/templates/acl_groups.html`
- Modify: `src/admin/routes.py` — add ACL group endpoints
- Modify: `src/admin/templates/base.html` — add nav link

### Phase 2
- Create: `src/auth/ldap.py`
- Modify: `src/admin/routes.py` — update login flow
- Modify: `src/config.py` — add LDAP settings
- Modify: `Dockerfile` — add LDAP deps (if using python-ldap; not needed for ldap3)

### Phase 3
- Create: `src/ingestion/cifs_scanner.py`
- Create: `src/admin/templates/cifs_connectors.html`
- Modify: `src/db/models.py` — add CIFSConnector model
- Modify: `src/db/metadata.py` — add CIFS connector CRUD + migration
- Modify: `src/admin/routes.py` — add CIFS connector endpoints
- Modify: `src/admin/templates/base.html` — add nav link
- Modify: `src/config.py` — add CIFS settings
- Modify: `Dockerfile` — add Kerberos client libs
- Modify: `docker-compose.yml` — add keytab/krb5.conf volume mounts
