# Release / GitHub publish checklist

## Before push

- [ ] No `.env`, `data/settings.json`, or real API keys in the commit  
- [ ] `data/` runtime (LanceDB, metadata.db, lightrag) is gitignored  
- [ ] Office lock files (`~$*`) removed  
- [ ] `pytest` (or at least auth + settings tests) passes  
- [ ] README and `docs/API_APPLICATIONS.md` match current Security UI  
- [ ] `.env.example` has placeholders only  
- [ ] `pip install -r requirements.lock.txt` then `pip-audit -r requirements.lock.txt` → **no known vulnerabilities**  
- [ ] `constraints-security.txt` floors still match current High+ advisories  

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

- [ ] Tag release if desired (`v0.x.y`)  
- [ ] Confirm GitHub Actions / secrets are not required for public clone + local run  
