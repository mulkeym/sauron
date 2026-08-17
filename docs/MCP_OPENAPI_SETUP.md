# Native MCP integration with OpenWebUI

Sauron exposes a native MCP Streamable HTTP endpoint from the same FastAPI
process and port as its REST API and Admin UI:

```text
https://sauron.example.internal/mcp
```

`mcpo`, a second MCP container, and ports 8090/8091 are not required. Native
MCP first appeared in OpenWebUI 0.6.31. This production configuration
requires OpenWebUI 0.9.6 or newer because that release fixed custom-header
template expansion for MCP connections; Sauron relies on `{{USER_GROUPS}}`
and `{{USER_EMAIL}}`.

## How OpenWebUI authenticates to Sauron MCP

OpenWebUI is the MCP **client**. Sauron never sees the user's OpenWebUI
password or login cookie. OpenWebUI is the source of truth for *who is
chatting* until a corporate IdP is connected.

There are two pieces on every call:

1. **Application key** (`X-API-Key`) — proves the caller is your OpenWebUI
   server (create this in Sauron **Settings → Security**).
2. **Logged-in OpenWebUI user** — OpenWebUI fills username and groups from
   that session. You do **not** request a Sauron JWT per user.

### Headers the MCP client must send

Put these on the OpenWebUI External Tools connection (custom headers).
Sauron will reject the call if they are missing.

| Header | Required | Value when setting up OpenWebUI | Purpose |
|---|---|---|---|
| `X-API-Key` | **Yes** | The Sauron application key for OpenWebUI | Identifies the MCP client |
| `X-Sauron-Username` | **Yes** (unless using a Sauron Bearer JWT) | `{{USER_EMAIL}}` | Who is asking |
| `X-Sauron-User-Groups` | Yes, for document ACL | `{{USER_GROUPS}}` | Which Sauron ACL groups apply |
| `Authorization` | **Do not set** in OpenWebUI | Leave Authentication as **None** | Reserved for Sauron JWTs on scripts |
| `X-OpenWebUI-User-Jwt` | No | OpenWebUI may add this on its own | Ignored by Sauron for now |
| `X-OpenWebUI-Chat-Id` or `X-Session-Id` | No | `{{CHAT_ID}}` if the UI offers it | Groups Switchyard LLM calls |

OpenWebUI expands `{{USER_EMAIL}}`, `{{USER_GROUPS}}`, and `{{CHAT_ID}}`
from the signed-in user on each request.

A non-OpenWebUI MCP client sends `X-API-Key` plus
`Authorization: Bearer <Sauron JWT>` instead of the two `X-Sauron-*`
headers. See **Direct Sauron MCP clients**.

```text
User logs into OpenWebUI  (local account)
        │
        ▼
User asks a question that uses a Sauron tool
        │
        ▼
OpenWebUI backend (not the browser) POSTs /mcp
        │  X-API-Key: <openwebui's Sauron app key>
        │  X-Sauron-Username: alice@corp   ← {{USER_EMAIL}}
        │  X-Sauron-User-Groups: finance,executives  ← {{USER_GROUPS}}
        ▼
Sauron checks the API key, then trusts those headers
        │
        ▼
Tools run as alice@corp with ACL groups finance + executives
```

### What you configure once in OpenWebUI

**Container env** (so templates expand):

```bash
ENABLE_FORWARD_USER_INFO_HEADERS=true
WEBUI_SECRET_KEY=<persistent-openwebui-secret>
```

**Admin → External Tools → Sauron** (MCP Streamable HTTP):

| Setting | Value |
|---|---|
| Authentication | **None** |
| URL | `http://<sauron-host>:8880/mcp` |

Custom headers:

```json
{
  "X-API-Key": "<openwebui application key>",
  "X-Sauron-Username": "{{USER_EMAIL}}",
  "X-Sauron-User-Groups": "{{USER_GROUPS}}"
}
```

`Authentication: None` is correct. Do **not** put OpenWebUI's own session
token in `Authorization: Bearer` — Sauron would treat that as a Sauron JWT
and reject it.

Create OpenWebUI **groups** whose names match Sauron ACL groups (`finance`,
`engineering`, …) and add users to them. A user with no matching groups can
call tools but sees no protected documents.

`{{USER_EMAIL}}` and `{{USER_GROUPS}}` are filled by OpenWebUI on each
request from the signed-in user. No per-user token minting.

OpenWebUI may also send `X-OpenWebUI-User-Jwt`. Sauron **ignores** it for
now (paused until an IdP). Identity is the two `X-Sauron-*` headers.

### What Sauron does with that request

```text
1. X-API-Key valid?     no → 403
2. X-Sauron-Username set?  no (and no Sauron Bearer) → 401
3. Groups from X-Sauron-User-Groups; strip ALL unless opted in
4. Run the tool with that username + groups
```

Anyone who has the OpenWebUI application key can send any username and
groups. Keep the key only on the OpenWebUI host and keep `/mcp` off the
public internet.

Sauron's `ALL` group is superuser. It is stripped from forwarded groups
unless `MCP_OPENWEBUI_ALLOW_ALL_GROUP=true`.

### Other MCP clients (not OpenWebUI)

Scripts and other apps should not reuse OpenWebUI's key. Give them their
own key and a Sauron JWT from `POST /api/v1/auth/token` (lab only). See
**Direct Sauron MCP clients** below.

## 1. Configure Sauron

Set these environment variables on the Sauron container. Use either a
DB-backed application key created in **Settings -> Security**, or use
`API_KEYS` only as a bootstrap/legacy allowlist:

```bash
MCP_ENABLED=true
MCP_PATH=/mcp
MCP_STATELESS_HTTP=true
# Bootstrap alternative to a DB-backed application key:
API_KEYS=<dedicated-sauron-application-key>
MCP_OPENWEBUI_USERNAME_HEADER=X-Sauron-Username
MCP_OPENWEBUI_GROUPS_HEADER=X-Sauron-User-Groups
MCP_OPENWEBUI_ALLOW_ALL_GROUP=false
```

In Kubernetes, put `API_KEYS` in a Secret, not a ConfigMap or Helm values
file committed to source control. `MCP_OPENWEBUI_JWT_SECRET` is unused while
OpenWebUI JWT verification is paused.

The same Sauron application-key management used by the REST API applies here.
A DB-backed, dedicated OpenWebUI application key is preferred because it can be
revoked and its use is recorded.

Changing `MCP_ENABLED` or `MCP_PATH` requires a Sauron restart because the MCP
ASGI application is mounted during process startup.

## 2. Configure OpenWebUI identity forwarding

Set these environment variables on OpenWebUI:

```bash
ENABLE_FORWARD_USER_INFO_HEADERS=true
WEBUI_SECRET_KEY=<persistent-openwebui-secret>
```

`ENABLE_FORWARD_USER_INFO_HEADERS` is required so OpenWebUI expands
`{{USER_EMAIL}}` and `{{USER_GROUPS}}` on the Sauron connection. The signed
`X-OpenWebUI-User-Jwt` path is paused; `FORWARD_USER_INFO_HEADER_JWT_SECRET` is
optional until an IdP is added. Keep `WEBUI_SECRET_KEY` persistent across
restarts.

### Start the OpenWebUI Docker container

The repository includes `openwebui.sh` for a single-container deployment. The
script uses the `open-webui` Docker volume to keep the OpenWebUI database.

1. Create `.openwebui.env` in the repository root:

   ```dotenv
   ENABLE_FORWARD_USER_INFO_HEADERS=true
   WEBUI_SECRET_KEY=<persistent-openwebui-secret>
   ```

2. Restrict access to the file:

   ```bash
   chmod 600 .openwebui.env
   ```

3. Start or replace the `open-webui` container:

   ```bash
   ./openwebui.sh
   ```

The script removes the current `open-webui` container before it creates the
new container. It does not remove the `open-webui` Docker volume.

Set `OPENWEBUI_ENV_FILE` to use an environment file at a different path:

```bash
OPENWEBUI_ENV_FILE=/secure/path/openwebui.env ./openwebui.sh
```

## 3. Add Sauron as an OpenWebUI tool server

### Automated Docker setup (Open WebUI v0.11+)

Some Open WebUI builds show only the browser-direct OpenAPI integration screen.
Their backend can still support native MCP. For a Docker deployment, the setup
script updates the stored admin-proxied connection. It preserves other tool
servers and creates a timestamped database backup.

```bash
export SAURON_API_KEY='<dedicated-sauron-application-key>'
scripts/configure_openwebui_sauron_mcp.sh \
  --container open-webui \
  --url http://192.168.1.181:8880/mcp
```

To also create matching Open WebUI groups and assign one user:

```bash
scripts/configure_openwebui_sauron_mcp.sh \
  --container open-webui \
  --url http://192.168.1.181:8880/mcp \
  --user user@example.com \
  --groups engineering,finance,contracts
```

Pass the API key through `SAURON_API_KEY`, or use the hidden prompt. Do not put
the key on the command line. Complete the identity configuration in section 2
before you run the setup script. You can run the script again to update the
Sauron connection. The script does not change other connections.

In **Admin Settings -> External Tools**, add a server with:

| Setting | Value |
|---|---|
| Type | MCP (Streamable HTTP) |
| URL | `https://<sauron-host>/mcp` |
| Authentication | None |
| Access Control | Only approved OpenWebUI groups/users |

Set the connection's custom headers. These three are required:

| Header | Value |
|---|---|
| `X-API-Key` | Sauron application key for this OpenWebUI |
| `X-Sauron-Username` | `{{USER_EMAIL}}` |
| `X-Sauron-User-Groups` | `{{USER_GROUPS}}` |

```json
{
  "X-API-Key": "<dedicated-sauron-application-key>",
  "X-Sauron-Username": "{{USER_EMAIL}}",
  "X-Sauron-User-Groups": "{{USER_GROUPS}}"
}
```

Optional: if the OpenWebUI build can template a chat or session id, add
`X-Session-Id` or `X-OpenWebUI-Chat-Id` so Sauron reuses that id on every
upstream LLM call for the answer (Switchyard live view / stats). Without it,
Sauron mints one UUID per question. `X-Sauron-Username` is used as the
agent id when no Switchyard / OpenWebUI user-id header is present.

Authentication is set to `None` because the application key and identity
headers are carried as custom headers.

OpenWebUI group names must exactly match the ACL group names assigned to Sauron
documents. A user with no matching groups receives no document results.

## 4. Run:AI, Knative, and proxy settings

Only container port 8080 needs to be published. The existing route for the Admin
UI and REST API also carries `/mcp`; no path split or second service is needed.

MCP is configured as stateless HTTP so initialization and tool calls can be
served safely without session affinity. Sauron currently uses JSON responses,
which avoids SSE buffering requirements, but research calls can remain open for
several minutes. Set the Knative/ingress request and upstream read timeouts above
the longest allowed Sauron query duration (normally at least 600 seconds).

If the platform publishes Sauron beneath a URL prefix, confirm that it preserves
or rewrites `/mcp` consistently. `MCP_PATH` controls the path inside Sauron.

## 5. Verification

First verify the shared application route remains healthy:

```bash
curl -fsS https://<sauron-host>/api/health
```

Then use OpenWebUI's **Verify Connection** action. Sauron should log an MCP
initialize request, and OpenWebUI should discover tools such as
`tool_ask`, `tool_search_documents`, and `tool_list_documents`.

Test with two users in different groups. Each user should see only documents
whose Sauron ACL intersects their forwarded OpenWebUI groups. This cross-group
negative test is required before production release.

Expected outcomes:

| Test | Expected result |
|---|---|
| Missing/invalid `X-API-Key` | HTTP 403 |
| Missing `X-Sauron-Username` and no Sauron Bearer | HTTP 401 |
| Invalid/expired Sauron Bearer JWT | HTTP 401 (no header fallback) |
| Valid identity with no matching groups | Tool succeeds with no protected documents |
| OpenWebUI group named `ALL` | Removed unless `MCP_OPENWEBUI_ALLOW_ALL_GROUP=true` |

## Troubleshooting

- **OpenWebUI cannot verify the connection:** confirm it is 0.9.6+, the server
  type is **MCP (Streamable HTTP)**, and the URL ends in `/mcp`.
- **401 from Sauron:** send `X-Sauron-Username` (and usually
  `X-Sauron-User-Groups`) after a valid API key, or a Sauron Bearer JWT.
  Enable `ENABLE_FORWARD_USER_INFO_HEADERS` so OpenWebUI expands
  `{{USER_EMAIL}}` / `{{USER_GROUPS}}`.
- **403 from Sauron:** verify the dedicated application key is active and is
  sent as `X-API-Key` rather than as a bearer token.
- **Tools appear but return no documents:** ensure `{{USER_GROUPS}}` is present
  in the custom header and OpenWebUI group names exactly match Sauron ACL names.
- **Long calls end at the proxy:** increase the Knative/ingress request and
  upstream read timeout to at least the longest permitted Sauron query.

## Direct Sauron MCP clients

See **Authentication for MCP clients** above. Short form:

```text
X-API-Key: <this-client's-application-key>
Authorization: Bearer <token from POST /api/v1/auth/token>
```

The user groups embedded in the Sauron JWT are used for ACL filtering. The
legacy stdio and SSE runners are not part of the production deployment because
they cannot carry this HTTP request identity safely.

## Upstream references

- [OpenWebUI native MCP configuration](https://docs.openwebui.com/features/extensibility/mcp/)
- [OpenWebUI identity-forwarding environment settings](https://docs.openwebui.com/reference/env-configuration/)
