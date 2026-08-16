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

## Authentication for MCP clients (OAuth-shaped)

Sauron treats every MCP call the way an OAuth resource server treats an API
call: **a confidential client** plus **a resource owner**. Those are two
different credentials. Neither one is the admin UI cookie.

| OAuth idea | Sauron MCP today | After an IdP |
|---|---|---|
| Confidential client (`client_id` + `client_secret`) | Application API key in `X-API-Key` | Same key (or a client credential from the IdP) |
| Authorization server | OpenWebUI local user/group store | Entra / Okta / Keycloak / … |
| Access token (who is the user, what groups) | Trusted headers from OpenWebUI, or a Sauron-minted JWT for scripts | Bearer access token issued by the IdP |
| Resource server | Sauron `/mcp` | Same |

The API key answers *“is this OpenWebUI (or another registered app)?”*  
The user credential answers *“which person, and which ACL groups?”*

Document access is **never** granted by the API key alone. Tools filter on
the user groups from the second credential.

### What happens on each `POST /mcp`

```text
1. Validate X-API-Key          → 403 if missing or unknown
2. Identify the user:
     a. Authorization: Bearer <Sauron JWT>   (scripts / direct clients)
     b. else X-Sauron-Username + X-Sauron-User-Groups
        (OpenWebUI; trusted because step 1 succeeded)
     c. else 401 Missing user identity
3. Strip group ALL unless MCP_OPENWEBUI_ALLOW_ALL_GROUP=true
4. Run the MCP method with that username + groups
```

An invalid or expired **Sauron** Bearer token is 401. It does **not** fall
through to the username header.

`X-OpenWebUI-User-Jwt` is **paused** until the IdP work lands. OpenWebUI may
still send it; Sauron ignores it.

### OpenWebUI (current source of truth)

Users log into OpenWebUI with its local accounts. OpenWebUI is the
authorization server for that session. When it calls Sauron it expands
templates from the logged-in user — you do not mint a token per user:

```json
{
  "X-API-Key": "<openwebui application key>",
  "X-Sauron-Username": "{{USER_EMAIL}}",
  "X-Sauron-User-Groups": "{{USER_GROUPS}}"
}
```

Set the MCP connection **Authentication** to **None**. The key and identity
travel as custom headers, not as OpenWebUI’s own login JWT (that JWT is not
a Sauron token and must not be sent as `Authorization: Bearer`).

Enable `ENABLE_FORWARD_USER_INFO_HEADERS=true` so `{{USER_EMAIL}}` and
`{{USER_GROUPS}}` expand. OpenWebUI group names must match Sauron ACL group
names (`finance`, `engineering`, …).

**Trust model:** anyone who has the OpenWebUI application key can send any
username and any groups. That is acceptable only while the key lives solely
on the OpenWebUI host and `/mcp` is not public. Treat the key like an OAuth
client secret.

### Direct clients (scripts, curl, a second app)

These use Sauron as a tiny authorization server for the lab:

```http
POST /api/v1/auth/token
{"username": "mike", "password": "…", "groups": ["finance"]}
```

Then:

```http
POST /mcp
X-API-Key: <that client's own application key>
Authorization: Bearer <access_token from /auth/token>
```

Groups come from the JWT, not from headers. `/auth/token` does **not**
check a real password in this lab path — the caller chooses username and
groups. Do not expose that mint to the internet; replace it with the IdP
when you are ready.

Give each client its **own** application key. Do not reuse OpenWebUI’s key.

### Planned IdP / OAuth path

When the corporate IdP is available, the shape stays the same and only the
**user** credential changes:

1. User signs in at the IdP (or at OpenWebUI via OIDC).
2. The client sends `Authorization: Bearer <IdP access token>`.
3. Sauron validates issuer, signature, audience, and expiry (like any OAuth
   resource server).
4. Username and groups come from token claims (`sub` / `email`, `groups` or
   `roles`), mapped to Sauron ACL names.
5. `X-API-Key` can remain as the confidential-client check, or be replaced
   by an OAuth client-credentials flow.

Until that lands, do not share `MCP_OPENWEBUI_JWT_SECRET` with new clients
and do not treat `/api/v1/auth/token` as production identity.

Sauron's `ALL` group grants unrestricted document access. It is stripped from
forwarded OpenWebUI groups by default. Do not enable
`MCP_OPENWEBUI_ALLOW_ALL_GROUP` unless that behavior is explicitly required.

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

Set the connection's custom Headers JSON to:

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
