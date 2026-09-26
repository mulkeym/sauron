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

## Phase 2: Wolfi base image (removes the no-fix findings)

The Debian image still carried 1 Critical, 108 High and 193 Medium findings
with no Debian fix (libxml2, util-linux, tesseract, poppler/libcurl, GnuPG, and
inkscape's dependency tree: a second Python 3.13, CUPS, Avahi, systemd, X11).
Probe scans of the same package set: Debian 13 1/110/199, Debian without
inkscape 1/92/120, **Wolfi 0/0/0**. Alpine was not used: it is musl-based,
and CPU PyTorch, onnxruntime, OpenCV and LanceDB ship glibc (manylinux) wheels
only. Wolfi is glibc-based, so every Python wheel installs unchanged.

What changed:

- All stages build `FROM ${WOLFI_IMAGE}` (default
  `cgr.dev/chainguard/wolfi-base`, pinned by digest; bump it periodically and
  mirror it for air-gapped builds). The runtime stage runs `apk upgrade`.
  Python is Wolfi's `python-3.11`; system packages come from `apk`.
- Wolfi does not package Inkscape or libvisio-tools, so a `native-tools`
  stage builds them from pinned sources (`docker/native-sources.sha256`):
  libsigc++ 2.12.1, glibmm 2.66.10, cairomm 1.14.6, pangomm 2.46.5,
  atkmm 2.28.5, gtkmm 3.24.11, Inkscape 1.4.4 (same 1.4 series as Debian's
  1.4.0; optional importers, spellcheck, D-Bus and NLS disabled), librevenge
  0.0.5 and libvisio 0.1.8 (`vsd2xhtml`; built as C++17 for ICU 78). The stage
  records the sonames the binaries link, and the runtime installs exactly those
  via `apk add so:...`.
- DejaVu 2.37 fonts (Debian's default sans-serif) are installed from the
  pinned upstream release so Visio text measurement matches the Debian image.
- Base-image Python tooling is removed from `/usr/lib/python3.11`
  (`check_packaged_runtime.sh` now finds it through `sys.base_prefix`).
- The corporate CA script works unchanged with Wolfi's `update-ca-certificates`.
- Image size drops from 11 GB to 7.5 GB.

**Scanner blind spot:** the source-built components above are not in any
package database, so Trivy and Harbor cannot report CVEs for them. Track
Inkscape, the gtkmm stack, librevenge, libvisio and DejaVu releases manually
and bump the URLs and hashes in the Dockerfile and `docker/native-sources.sha256`
together. First builds compile for roughly an hour; the CI timeout is 240 min.

## Phase 3: non-root runtime (Trivy DS-0002)

- The image ends with `USER 65532:65532` (Wolfi's standard `nonroot` user,
  `HOME=/home/nonroot`).
- Baked models moved from `/root/.cache` to `/opt/models` (`HF_HOME`,
  `SENTENCE_TRANSFORMERS_HOME`); they, the code and `/opt/venv` stay
  root-owned and read-only. Only `/app/data`, `/tmp` and `$HOME` are writable.
- Docker Compose: a one-shot `data-permissions` service (root, no network,
  only `CHOWN`/`DAC_OVERRIDE`/`FOWNER`) re-owns a volume written by an older
  root image, then exits; later starts change nothing. The `api` service waits
  for it, drops all capabilities and sets `no-new-privileges`. Both use
  `${SAURON_IMAGE:-sauron:local}`.
- Helm: `runAsUser/runAsGroup/fsGroup: 65532`, `runAsNonRoot`,
  `fsGroupChangePolicy: OnRootMismatch`, RuntimeDefault seccomp, no privilege
  escalation, all capabilities dropped.
- The entrypoint fails fast with instructions if `/app/data` is not writable.
- `check_packaged_runtime.sh` asserts the non-root uid, writable data,
  read-only models/code, and loads the embedding model offline as that user.
  CI runs the regression suite as the unprivileged user too.

Still open (outside the report): build-arg handling of `HF_TOKEN` (DS-0031) —
see Phase 2 of [CONTAINER_CVE_REMEDIATION_PLAN.md](CONTAINER_CVE_REMEDIATION_PLAN.md).

## Verification, Debian phase (26 Sep 2026, linux/amd64)

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

## Verification, Wolfi phase (26 Sep 2026, linux/amd64)

- Trivy 0.74.0: **0 findings of any severity** (151 Wolfi packages and 265
  Python packages inventoried). None of the report's 52 IDs are present.
- `scripts/check_packaged_runtime.sh`: passed.
- Full test suite inside the image, offline: 1399 passed, 1 skipped (includes
  the EMF rendering tests through Inkscape and the Visio tests through
  `vsd2xhtml`).

## Verification, non-root phase (26 Sep 2026, linux/amd64)

- Trivy 0.74.0: 0 vulnerabilities; `trivy config Dockerfile`: 27 checks,
  0 failures (DS-0002 root user and DS-0001 untagged base both cleared).
- `scripts/check_packaged_runtime.sh`: passed, including uid != 0, writable
  `/app/data`, read-only models/code/venv, and an offline Nomic embedding load
  as the unprivileged user.
- Full test suite as uid 65532: 1399 passed, 1 skipped.
  `test_hung_child_times_out_without_blocking_event_loop` is timing-sensitive:
  it fails on hosts with coarse timers (an idle 50 ms asyncio sleep measured
  130-200 ms on one Docker host) regardless of user, and passes on the Proxmox
  test container.
- In-place upgrade of a deployment whose volume was written by the root image:
  `data-permissions` re-owned it once (0 files left with another owner), a
  second run changed nothing, existing documents remained available, and
  ingestion, knowledge-graph insertion, cited answers and spreadsheet queries
  worked as uid 65532.
