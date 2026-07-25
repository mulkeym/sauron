# Application API Keys

SAURON supports **multiple service clients** (demo front-ends, OpenWebUI, MCP gateways, internal tools). Each client should use its own application API key.

## Concepts

| Concept | Role |
|---------|------|
| **Application** | Named client (`sdwan-demo-chat`, `openwebui`, …) |
| **API key** | Service secret for that client (`X-API-Key` header) |
| **JWT** | User identity + ACL **groups** (document access) |

Keys answer: *“Is this a trusted backend?”*  
JWT groups answer: *“Which documents may this request see?”*

## Admin UI

**Settings → Security → Applications & API Keys**

1. **Add application** — slug, display name, description  
2. **Generate key** — copy the full secret **once** (only a prefix is stored/shown later)  
3. **Revoke** a key or **deactivate** an entire app without affecting others  

Keys are stored as **SHA-256 hashes** in SQLite (`api_applications`, `api_key_records`).

## Migration from `API_KEYS` env

On startup, any secrets listed in `API_KEYS` / admin settings string that are not already in the DB are imported under application **`legacy`**.  

Prefer moving each real client to its own application and revoking unused legacy keys when ready.

## Example: register a demo backend

1. Create application `sdwan-demo-chat`  
2. Generate key → set on the demo server as `SAURON_API_KEY`  
3. Demo backend mints JWTs per persona and calls `/api/v1/query` or `/api/v1/query/async`  

The **browser must not** hold Sauron API keys when deploying multi-user front-ends.

## Validation order

1. Match key hash against active, non-revoked DB keys for an **active** application  
2. Else allow legacy plaintext list from `settings.api_keys` / env (migration)  

`last_used_at` is updated best-effort on successful DB key auth.
