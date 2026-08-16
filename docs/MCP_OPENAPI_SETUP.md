# Native MCP integration with OpenWebUI

Sauron exposes a native MCP Streamable HTTP endpoint from the same FastAPI
process and port as its REST API and Admin UI:

```text
https://sauron.example.internal/mcp
```

`mcpo`, a second MCP container, and ports 8090/8091 are not required. Native
MCP first appeared in OpenWebUI 0.6.31. This production configuration
requires OpenWebUI 0.9.6 or newer because that release fixed custom-header
template expansion for MCP connections; Sauron relies on `{{USER_GROUPS}}`.

## Request and trust flow

```text
OpenWebUI user
   -> OpenWebUI MCP client
      -> Run:AI / Knative HTTPS route
         -> Sauron :8080/mcp
            -> application API-key validation
            -> signed OpenWebUI identity validation
            -> Sauron ACL filtering using forwarded OpenWebUI group names
```

Every MCP request must contain both:

1. A dedicated Sauron application key in `X-API-Key`.
2. User identity. Production OpenWebUI deployments use the short-lived,
   HS256-signed `X-OpenWebUI-User-Jwt` header.

Sauron verifies the OpenWebUI token's HS256 signature, `iss=open-webui`, `sub`,
`iat`, and `exp` claims. A normal OpenWebUI login/session token sent as
`Authorization: Bearer` is not accepted: that header is reserved for JWTs
issued by Sauron for direct clients.

OpenWebUI's signed identity JWT does not contain group claims. The MCP
connection therefore sends the current user's group names in
`X-Sauron-User-Groups`. Sauron trusts that header only after validating the
dedicated application key and signed user identity. Keep the application key
exclusive to OpenWebUI and restrict `/mcp` to trusted network paths.

Sauron's `ALL` group grants unrestricted document access. It is stripped from
OpenWebUI-forwarded groups by default. Do not enable
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
MCP_OPENWEBUI_JWT_SECRET=<shared-random-secret>
MCP_OPENWEBUI_GROUPS_HEADER=X-Sauron-User-Groups
MCP_OPENWEBUI_ALLOW_ALL_GROUP=false
```

Use a secret of at least 32 random bytes. In Kubernetes, put
`API_KEYS` and `MCP_OPENWEBUI_JWT_SECRET` in a Secret, not a ConfigMap or Helm
values file committed to source control.

The same Sauron application-key management used by the REST API applies here.
A DB-backed, dedicated OpenWebUI application key is preferred because it can be
revoked and its use is recorded.

Changing `MCP_ENABLED` or `MCP_PATH` requires a Sauron restart because the MCP
ASGI application is mounted during process startup.

## 2. Configure OpenWebUI identity forwarding

Set these environment variables on OpenWebUI:

```bash
ENABLE_FORWARD_USER_INFO_HEADERS=true
FORWARD_USER_INFO_HEADER_JWT_SECRET=<same-shared-random-secret>
WEBUI_SECRET_KEY=<persistent-openwebui-secret>
```

`FORWARD_USER_INFO_HEADER_JWT_SECRET` must exactly match Sauron's
`MCP_OPENWEBUI_JWT_SECRET`. Keep `WEBUI_SECRET_KEY` persistent across restarts
and common to all OpenWebUI replicas.

The default forwarded header name is `X-OpenWebUI-User-Jwt`. If OpenWebUI's
`FORWARD_USER_INFO_HEADER_JWT` setting is customized, Sauron and any intervening
proxy must be configured to preserve that name; the current Sauron integration
expects the default header.

### Start the OpenWebUI Docker container

The repository includes `openwebui.sh` for a single-container deployment. The
script uses the `open-webui` Docker volume to keep the OpenWebUI database.

1. Create `.openwebui.env` in the repository root:

   ```dotenv
   ENABLE_FORWARD_USER_INFO_HEADERS=true
   FORWARD_USER_INFO_HEADER_JWT_SECRET=<same-shared-random-secret>
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
  "X-Sauron-User-Groups": "{{USER_GROUPS}}"
}
```

Optional: if the OpenWebUI build can template a chat or session id, add
`X-Session-Id` or `X-OpenWebUI-Chat-Id` so Sauron reuses that id on every
upstream LLM call for the answer (Switchyard live view / stats). Without it,
Sauron mints one UUID per question. `X-OpenWebUI-User-Id` (or the signed JWT
`sub`) is used as `x-switchyard-agent-id`.

Authentication is set to `None` because the application key is carried in the
custom header and OpenWebUI forwards the signed identity JWT separately.

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
| Missing user identity | HTTP 401 |
| Invalid/expired OpenWebUI identity JWT | HTTP 401 |
| Valid identity with no matching groups | Tool succeeds with no protected documents |
| OpenWebUI group named `ALL` | Removed unless `MCP_OPENWEBUI_ALLOW_ALL_GROUP=true` |

## Troubleshooting

- **OpenWebUI cannot verify the connection:** confirm it is 0.9.6+, the server
  type is **MCP (Streamable HTTP)**, and the URL ends in `/mcp`.
- **401 from Sauron:** enable `ENABLE_FORWARD_USER_INFO_HEADERS`, confirm the
  two forwarding secrets match exactly, and restart OpenWebUI.
- **403 from Sauron:** verify the dedicated application key is active and is
  sent as `X-API-Key` rather than as a bearer token.
- **Tools appear but return no documents:** ensure `{{USER_GROUPS}}` is present
  in the custom header and OpenWebUI group names exactly match Sauron ACL names.
- **Long calls end at the proxy:** increase the Knative/ingress request and
  upstream read timeout to at least the longest permitted Sauron query.

## Direct Sauron MCP clients

Non-OpenWebUI clients may use Sauron's existing signed user JWT instead:

```text
X-API-Key: <sauron-application-key>
Authorization: Bearer <sauron-user-jwt>
```

The user groups embedded in the Sauron JWT are used for ACL filtering. The
legacy stdio and SSE runners are not part of the production deployment because
they cannot carry this HTTP request identity safely.

## Upstream references

- [OpenWebUI native MCP configuration](https://docs.openwebui.com/features/extensibility/mcp/)
- [OpenWebUI identity-forwarding environment settings](https://docs.openwebui.com/reference/env-configuration/)
