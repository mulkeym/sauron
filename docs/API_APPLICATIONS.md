# Application API Keys

SAURON supports **multiple service clients** (demo front-ends, OpenWebUI, MCP gateways, internal tools). Each client should use its own application API key.

## Concepts

| Concept | Role |
|---------|------|
| **Application** | Named client (`sdwan-demo-chat`, `openwebui`, …) |
| **API key** | Service secret for that client (`X-API-Key` header) |
| **User identity** | Trusted headers from OpenWebUI, or a Sauron JWT for direct clients |

Keys answer: *“Is this OpenWebUI (or another registered app)?”*  
User identity answers: *“Which OpenWebUI user, which documents?”*

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

## LLM session headers (Switchyard)

Answer-pipeline LLM calls that Sauron makes upstream (classify, SQL, MAP,
synthesize, LightRAG query) share one Switchyard session. Clients may pass a
conversation id so follow-up questions stay grouped:

| Inbound header | Used as |
|----------------|---------|
| `x-switchyard-session-id` | Session id (highest precedence) |
| `x-session-id` | Session id |
| `session-id` | Session id |
| `x-openwebui-chat-id` | Session id |
| `x-switchyard-agent-id` | Stable user/agent id |
| `x-openwebui-user-id` | Agent id if the Switchyard header is absent |

If none of the session headers are set, Sauron mints a UUID for that question
only. Demo chat and OpenWebUI should forward their chat/session id when they
have one. See the [session-headers design](superpowers/specs/2026-08-15-llm-session-headers-design.md).

## OpenWebUI native MCP application

Create a dedicated application such as `openwebui-mcp` and generate a key for
the OpenWebUI External Tools connection. OpenWebUI sends that key only from its
backend:

```json
{
  "X-API-Key": "<openwebui-mcp-application-key>",
  "X-Sauron-Username": "{{USER_EMAIL}}",
  "X-Sauron-User-Groups": "{{USER_GROUPS}}"
}
```

The application key proves the request came from your OpenWebUI server. It
does not grant document access. OpenWebUI is the source of truth for the
person chatting: it expands `{{USER_EMAIL}}` and `{{USER_GROUPS}}` from the
logged-in user. Sauron ignores `X-OpenWebUI-User-Jwt` for now.

See [MCP_OPENAPI_SETUP.md](MCP_OPENAPI_SETUP.md#how-openwebui-authenticates-to-sauron-mcp)
for the OpenWebUI login → header → ACL walkthrough.

## Validation order

1. Match key hash against active, non-revoked DB keys for an **active** application  
2. Else allow legacy plaintext list from `settings.api_keys` / env (migration)  

`last_used_at` is updated best-effort on successful DB key auth.
