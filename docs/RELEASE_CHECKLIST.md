# Release / GitHub publish checklist

## Before push

- [ ] No `.env`, `data/settings.json`, or real API keys in the commit  
- [ ] `data/` runtime (LanceDB, metadata.db, lightrag) is gitignored  
- [ ] Office lock files (`~$*`) removed  
- [ ] `pytest` (or at least auth + settings tests) passes  
- [ ] README and `docs/API_APPLICATIONS.md` match current Security UI  
- [ ] Native MCP documentation matches `/mcp` on API port 8080; no production `mcpo`, SSE runner, or 8090/8091 service remains
- [ ] `.env.example` has placeholders only  
- [ ] `pip install -r requirements.lock.txt` then `pip-audit -r requirements.lock.txt` → **no known vulnerabilities**  
- [ ] `constraints-security.txt` floors still match current High+ advisories  
- [ ] Review the [container CVE remediation plan](CONTAINER_CVE_REMEDIATION_PLAN.md), retain a full Trivy report for the immutable image digest, and confirm there are no new fixable Critical/High findings
- [ ] `pytest -q tests/test_mcp` passes, including native HTTP auth and ACL propagation tests
- [ ] `docker compose config --quiet` and `helm lint charts/sauron` pass

## Local cleanup (optional)

```bash
# Safe local junk (does not delete your real index unless you choose)
rm -f ~$*.docx data/test_metadata.db
# Full re-seed of lab data only if you intend to wipe local RAG data:
# rm -rf data/lancedb data/lightrag data/metadata.db data/settings.json
```

## Suggested commit groups

1. **feat(auth):** application API keys (models, store, admin UI, tests)  
2. **feat:** CORS for local browser demos; docs  
3. Other pending work (figures, etc.) as separate commits if unrelated  

## After push

- [ ] Confirm **Docker** GitHub Action succeeded (builds & pushes to GHCR)  
- [ ] Record the published image digest and associate it with its SBOM and Trivy report
- [ ] Image available: `ghcr.io/mulkeym/sauron:latest` (and `sha-…` tag)  
- [ ] Tag release if desired (`v0.x.y`) → also publishes `ghcr.io/mulkeym/sauron:0.x.y`  
- [ ] If the package is private, set visibility or document pull credentials for deploy  
- [ ] Confirm secrets are not required for public clone + local `docker compose` run  
- [ ] For OpenWebUI integration, confirm version 0.9.6+, persistent `WEBUI_SECRET_KEY`, and matching `FORWARD_USER_INFO_HEADER_JWT_SECRET` / `MCP_OPENWEBUI_JWT_SECRET`
- [ ] Verify `/mcp` with two OpenWebUI users in different groups; each must be unable to retrieve the other group's documents
