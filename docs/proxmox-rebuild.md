# Rebuild the integrated source on Proxmox

This source combines GitHub `master` at
[`d7732e4`](https://github.com/mulkeym/sauron/commit/d7732e4fd444110da895b6838bbdec891c27deed)
with the existing local Sauron improvements. It includes the dashboard's
persistent recent-query history, upstream authentication and LLM session
tracking, PDF extraction in a separate process inside the same container,
evidence and ACL improvements, and editable answer profiles and settings.

The integrated changes are included in this repository. GitHub Actions builds
and publishes `ghcr.io/mulkeym/sauron:latest` after a successful build of
`master`. You can use that image, build from this checkout, or use the supplied
source archive. `BUILD_SOURCE.json` in the archive records its upstream commit
and SHA-256 hashes of the included source files.

## Prepare the Docker host

Run these commands inside the Linux VM or container on Proxmox where you run
Docker Compose. Clone the repository for a new deployment:

```bash
git clone https://github.com/mulkeym/sauron.git
cd sauron
```

Alternatively, transfer `sauron-proxmox-source.tar.gz` there and extract it into
a new directory:

```bash
mkdir sauron-integrated
tar -xzf sauron-proxmox-source.tar.gz -C sauron-integrated
cd sauron-integrated/sauron
```

Copy your existing deployment's `.env` and any required corporate CA file
into this directory. For a fresh deployment, start with `.env.example` and
configure the LLM URL/model, embedding settings, and `JWT_SECRET_KEY` before
starting. LLM credentials and the other application settings are also
available in the admin portal. Model downloads occur during the image build;
keep the model-prefetch steps enabled for an offline runtime.

The archive deliberately excludes credentials, databases, uploaded documents,
local model caches, and Git history. Transfer or restore your existing data
separately if moving from another host.

## Reuse the persistent data

Sauron's complete application data directory must remain mounted at
**`/app/data`**. The included Compose file uses:

```yaml
volumes:
  - app_data:/app/data
```

Reuse your current host-folder mapping or named volume. Compose prefixes
volume names with the project name, so extracting into a new directory can
otherwise select a new empty volume. To explicitly use an existing named
volume, create `compose.override.yml` with its actual name:

```yaml
volumes:
  app_data:
    external: true
    name: YOUR_EXISTING_VOLUME
```

Back up the existing data before upgrading. Preserve the existing Compose
project name when replacing a running deployment; use `-p YOUR_PROJECT`
with each command below if needed. This keeps Compose managing the existing
service instead of starting another one on the same port.

## Pull the GitHub image or build locally

After the GitHub **Docker** workflow succeeds, add this service override to
`compose.override.yml` (alongside your existing-volume override, if used):

```yaml
services:
  api:
    image: ghcr.io/mulkeym/sauron:latest
```

Then pull and start that image:

```bash
docker compose config --quiet
docker compose pull api
docker compose up -d --no-build api
docker compose ps
docker compose logs --tail=100 api
```

The GitHub image targets `linux/amd64`. For a local source build, leave the
service image as `sauron:local` and use:

```bash
docker compose config --quiet
docker compose build api
docker compose up -d api
docker compose ps
docker compose logs --tail=100 api
```

The default host port is **8880**. Open
`http://<docker-host>:8880/admin/`, sign in, and check the Dashboard and
Settings → Answer Profiles. After a query, the dashboard should show its
recent activity. The Docker health check uses `/admin/login`; direct calls
to `/api/health` now require an application API key.

## OpenWebUI migration

Keep the MCP connection at `http://<docker-host>:8880/mcp`, with
Authentication **None** and these custom headers:

```json
{
  "X-API-Key": "<Sauron application key>",
  "X-Sauron-Username": "{{USER_EMAIL}}",
  "X-Sauron-User-Groups": "{{USER_GROUPS}}"
}
```

Create the application key in Settings → Security, or configure `API_KEYS`
as a bootstrap value. Upstream removed the implicit development keys.

No shared JWT is required for this mode. In Settings → All Settings, enable
**Trust OpenWebUI user headers** if your saved configuration has it disabled.
Saved settings override environment defaults. Existing connections using
`X-OpenWebUI-User-Name` still work as a fallback to the default username
header. See [the MCP setup guide](MCP_OPENAPI_SETUP.md) for custom-header and
optional signed-identity configuration.

## Verification completed before packaging

- 864 tests passed, including PDF worker crash recovery, authentication,
  answer profiles, and query activity persisting across database reopening.
- Real offline Nomic embeddings and MiniLM reranking passed with the
  upstream pins `transformers==5.15.0` and `sentence-transformers==5.7.0`.
- Compose configuration, shell syntax, and merge whitespace checks passed.

These checks ran locally before publishing; no container build or deployment
was performed on Proxmox. The previously observed LightRAG startup-cleanup
compatibility warning is outside this integration and has not been changed.
