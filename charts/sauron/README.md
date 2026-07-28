# SAURON Helm Chart

Deploy [SAURON](https://github.com/mulkeym/sauron) (API + Admin UI, MCP server, MCP OpenAPI proxy) to Kubernetes / Run:ai.

This chart is designed for **self-hosted, air-gapped** clusters that pull images from **on-prem Harbor**.

## Chart location (for “Helm repo URL”)

If your platform asks for a **Helm / help repo URL**, point it at this GitHub repository and chart path:

| Field | Value |
|-------|--------|
| Repo URL | `https://github.com/mulkeym/sauron.git` |
| Chart path | `charts/sauron` |
| Branch | `master` (or your release branch) |

CLI install from Git (once the chart is on the remote):

```bash
helm upgrade --install sauron oci://...   # if you also publish OCI charts to Harbor
# or clone and install locally:
git clone https://github.com/mulkeym/sauron.git
helm upgrade --install sauron ./sauron/charts/sauron \
  -n sauron --create-namespace \
  -f ./sauron/charts/sauron/values-airgapped.yaml
```

## What gets deployed

| Component | Port | Description |
|-----------|------|-------------|
| **api** | 8080 | FastAPI REST API + Admin UI (`/admin`, `/api`) |
| **mcp** | 8090 | MCP server (SSE) |
| **mcpo** | 8091 | MCP → OpenAPI proxy for OpenWebUI / REST clients |

All three run as **containers in a single pod** sharing one PVC at `/app/data` (same model as `docker-compose.yml`). That keeps SQLite / LanceDB / DuckDB consistent without requiring ReadWriteMany storage.

Deployment strategy is `Recreate` so the RWO volume is never attached to two pods.

## Container image (GitHub builds this for you)

On every push to `master`/`main` and every `v*` tag, GitHub Actions builds the
Dockerfile and publishes to **GitHub Container Registry**:

| Tag | When |
|-----|------|
| `ghcr.io/mulkeym/sauron:latest` | Default branch |
| `ghcr.io/mulkeym/sauron:sha-<short>` | Every published build |
| `ghcr.io/mulkeym/sauron:1.2.3` | Git tag `v1.2.3` |

```bash
docker pull ghcr.io/mulkeym/sauron:latest
```

If the package is private, log in with a PAT that has `read:packages`:

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin
```

Chart defaults already point at GHCR (`image.registry=ghcr.io`, `image.repository=mulkeym/sauron`).

## Air-gapped prerequisites

1. **Get the image into Harbor** (pull from GHCR on a connected host, then push):

   ```bash
   docker pull ghcr.io/mulkeym/sauron:latest
   docker tag ghcr.io/mulkeym/sauron:latest harbor.example.local/library/sauron:1.0.0
   docker push harbor.example.local/library/sauron:1.0.0
   ```

   Or build from source if GHCR is unreachable:

   ```bash
   docker build -t harbor.example.local/library/sauron:1.0.0 .
   docker push harbor.example.local/library/sauron:1.0.0
   ```

   The Dockerfile already prefetches PDF/OCR models and sets `HF_HUB_OFFLINE=1` for offline runtime.

2. **Mirror base layers** into Harbor if builders cannot reach Docker Hub (`python:3.11-slim`, etc.).

3. **LLM endpoint** reachable from the cluster (vLLM or any OpenAI-compatible API). SAURON does not ship the LLM.

4. **StorageClass** available for the PVC (default `ReadWriteOnce`, 50–100Gi recommended).

## Install

### Minimal (dev)

```bash
helm upgrade --install sauron ./charts/sauron \
  -n sauron --create-namespace \
  --set image.registry=harbor.example.local \
  --set image.repository=library/sauron \
  --set image.tag=1.0.0 \
  --set config.vllmBaseUrl=http://vllm:8000/v1 \
  --set secrets.jwtSecretKey="$(openssl rand -hex 32)" \
  --set secrets.adminPassword='change-me'
```

### Air-gapped production-style

1. Copy `values-airgapped.yaml` and fill Harbor host, image tag, vLLM URL, storage class.
2. Prefer an externally managed Secret:

   ```bash
   kubectl create namespace sauron
   kubectl -n sauron create secret generic sauron-app-secrets \
     --from-literal=JWT_SECRET_KEY="$(openssl rand -hex 32)" \
     --from-literal=API_KEYS="app-key-1" \
     --from-literal=ADMIN_USERNAME=admin \
     --from-literal=ADMIN_PASSWORD='strong-password' \
     --from-literal=VLLM_API_KEY=''

   kubectl -n sauron create secret docker-registry harbor-pull-secret \
     --docker-server=harbor.example.local \
     --docker-username='robot$sauron-pull' \
     --docker-password='...'
   ```

3. Install:

   ```bash
   helm upgrade --install sauron ./charts/sauron \
     -n sauron \
     -f charts/sauron/values-airgapped.yaml \
     --set secrets.existingSecret=sauron-app-secrets \
     --set imagePullSecrets[0].name=harbor-pull-secret \
     --set imagePullSecretsCreate.enabled=false \
     --set image.registry=harbor.example.local \
     --set image.repository=library/sauron \
     --set image.tag=1.0.0
   ```

## Key values

| Path | Purpose |
|------|---------|
| `image.registry` / `repository` / `tag` / `digest` | Harbor image reference |
| `imagePullSecrets` / `imagePullSecretsCreate` | Harbor auth |
| `config.vllmBaseUrl` | OpenAI-compatible LLM base URL |
| `config.embeddingMode` | `local` (default, offline) or `api` |
| `persistence.*` | PVC for LanceDB + SQLite + DuckDB |
| `secrets.*` / `secrets.existingSecret` | JWT, API keys, admin password |
| `api` / `mcp` / `mcpo`.enabled | Toggle components |
| `ingress.*` | Optional external access |

See `values.yaml` for the full schema and `values-airgapped.yaml` for a Harbor-oriented example.

## Access

```bash
# Admin UI
kubectl port-forward -n sauron svc/sauron-api 8080:8080
# open http://localhost:8080/admin

# Health
kubectl exec -n sauron deploy/sauron -c api -- curl -sf http://localhost:8080/api/health
```

## Run:ai notes

- SAURON itself does **not** require GPUs when `embeddingMode=local` (CPU embeddings). GPU nodes are for your **vLLM / inference** workloads, which you point at via `config.vllmBaseUrl`.
- Give the API container enough memory (2–16Gi) if using local embeddings.
- Use your platform’s project/namespace isolation as usual; this chart only needs a namespace, a PVC, and pull access to Harbor.

## Packaging (optional Harbor OCI chart)

If your platform prefers an OCI Helm chart in Harbor rather than a Git URL:

```bash
helm package ./charts/sauron
helm push sauron-0.1.0.tgz oci://harbor.example.local/helm-charts
# Install:
helm upgrade --install sauron oci://harbor.example.local/helm-charts/sauron --version 0.1.0
```

## Uninstall

```bash
helm uninstall sauron -n sauron
# PVC is retained by default Kubernetes behavior only if you used retain policies;
# delete the PVC explicitly if you want to wipe data:
# kubectl -n sauron delete pvc -l app.kubernetes.io/instance=sauron
```
