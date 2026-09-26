# Cerberus remediation hand-off — September 2026

Source: Cerberus "Remediation hand-off" for `mulkeym/sauron:latest`
(Harbor scan 15 Sep 2026, generated 22 Sep 2026): 13 packages to upgrade
clearing 55 findings (52 unique IDs), plus 410 findings with no fix.

## Changes

| Report item | Change |
|---|---|
| perl-base, gzip, libc6/libc-bin, libpcre2-8-0, libsqlite3-0 (Critical/High/Medium) | `apt-get upgrade` in builder and runtime stages installs the Debian 13 security updates (`+deb13u*` versions in the report or later). Base pinned to `python:3.11-slim-trixie` (`PYTHON_IMAGE` build arg). |
| pip 24.0, setuptools 79.0.1, vendored jaraco.context 5.3.0 and wheel 0.45.1 (High/Medium) | These live in the base image's system Python, which Sauron never uses. The runtime stage removes system pip/setuptools/wheel and the stale `ensurepip` bundled wheels. |
| setuptools 70.3.0, msgpack 1.1.2 (High) — no file path | Come from `pip/_vendor/vendor.txt` inside pip itself (still present in pip 26.2.1). pip is uninstalled from `/opt/venv` after the build; the runtime never installs packages. `msgpack>=1.2.1`, `wheel>=0.46.2`, `jaraco.context>=6.1.0`, `pip>=26.2.0` floors added to `constraints-security.txt`; constraints now also apply to the CPU torch install. |
| libjpeg-turbo 1.5.3 (el8 RPM, High) | Not present in images built from this repository (Debian ships `libjpeg62-turbo` 2.1.5). Likely an attribution artifact of the Harbor/BuildKit SBOM for that digest; confirm with the next Cerberus import. |
| curl (no fix, High/Medium) | curl CLI removed; Docker health check and `scripts/check_packaged_runtime.sh` use Python's stdlib. `libcurl` remains because poppler and tesseract depend on it, so the libcurl findings remain. |

Related fix found during verification: SQLAlchemy 2.1 no longer installs
`greenlet` by default, so a fresh build failed to import the async metadata
store. `requirements.txt` now requires `sqlalchemy[asyncio]>=2.0.36,<2.1`.

CI (`docker-publish.yml`) now saves a full Trivy JSON report and fails the
build before publishing if any Critical/High finding has a fixed version.

## Remaining (no fix available)

The remaining Critical (`libxml2` CVE-2026-6653) and the High/Medium OS
findings have no Debian fix yet. They come from `util-linux`/`login`
(essential), tesseract, poppler/libcurl, GnuPG (pulled in by poppler's
gpgme), libtiff, libexpat, and libxml2 (inkscape, libvisio, librsvg). Removing
them would remove OCR, PDF rasterisation, or EMF/Visio rendering. They clear
automatically on rebuild once Debian publishes fixes.

Not in the report's scope but still open: Trivy config finding DS-0002
(container runs as root) and build-arg handling of `HF_TOKEN` (DS-0031) — see
Phase 2 of [CONTAINER_CVE_REMEDIATION_PLAN.md](CONTAINER_CVE_REMEDIATION_PLAN.md).

## Verification (26 Sep 2026, linux/amd64 build on the Proxmox node)

- Trivy 0.74.0 on the final image: **0 Critical/High/Medium findings with a
  fix available**. None of the 52 CVE/GHSA IDs in the report's "Packages to
  upgrade" table are present. Remaining totals (all unfixed upstream):
  1 Critical, 108 High, 193 Medium, 260 Low, 2 Unknown — versus 1 / 116 / 210
  / 272 for the published `ghcr.io/mulkeym/sauron:latest` scanned with the
  same database (4 High and 7 Medium of those were fixable).
- `scripts/check_packaged_runtime.sh`: passed (imports, native tools, offline
  models, no curl/system pip, anonymous `/api/health` 403, keyed 200).
- Full test suite inside the image, offline: 1399 passed, 1 skipped.
- End-to-end smoke test with local embeddings and vLLM
  (`gemma-4-26b-a4b-awq`): PDF, DOCX and XLSX ingestion; cited answer from the
  PDF; correct spreadsheet totals through the SQL path; Docker health check
  (Python stdlib) reports healthy.
