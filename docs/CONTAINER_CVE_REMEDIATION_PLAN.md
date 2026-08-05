# Container CVE Remediation Plan

Status: Backlog / not yet implemented
Review date: 2026-08-05
Scanner: Trivy 0.73.0
Scope: Published `linux/amd64` Sauron runtime image, Dockerfile configuration,
Python dependencies, Debian packages, and CI controls

## Purpose

This document records the August 2026 container security review and provides a
phased plan that can be actioned later. It is intentionally separate from an
individual Trivy database snapshot: vulnerability counts will change as the
database and base image change, but the remediation priorities and acceptance
criteria should remain stable.

No remediation described here was implemented as part of the review.

## Reviewed image

The final review used the immutable image produced from commit `03bb54b`:

```text
Tag:    ghcr.io/mulkeym/sauron:sha-03bb54b
Digest: ghcr.io/mulkeym/sauron@sha256:afbe7c810eb2c3d120f10d754c999daa83bbbbd9fa5d58bf1cd03cf088d464fd
OS:     Debian 13.6 (trixie), linux/amd64
```

The enterprise scan previously reported 6 Critical, 49 High, and 111 Medium.
The independent scan reproduced the same overall profile with a small
database/build-date delta:

| Package source | Critical | High | Medium | Findings with a fixed version |
|---|---:|---:|---:|---:|
| Debian OS packages | 6 | 42 | 99 | 0 |
| Python packages | 0 | 4 | 6 | 4 High, 6 Medium |
| **Total** | **6** | **46** | **105** | **4 High, 6 Medium** |

There were also 211 Low and 28 Unknown findings. These are retained in full
reports but are not the first remediation gate.

## Key findings

### 1. The Critical findings are inherited OS findings

The six Critical findings had no Debian fixed version at review time:

| Package | CVE | Notes |
|---|---|---|
| `perl-base` | CVE-2026-13221 | Inherited from Debian base |
| `perl-base` | CVE-2026-42496 | Debian status: fix deferred |
| `perl-base` | CVE-2026-57433 | Inherited from Debian base |
| `perl-base` | CVE-2026-8376 | Inherited from Debian base |
| `libglib2.0-0t64` | CVE-2026-58016 | Added with native image/PDF dependencies |
| `libxml2` | CVE-2026-6653 | Added with runtime native dependencies |

An official `python:3.11-slim-trixie` base scan produced 4 Critical, 21 High,
and 61 Medium findings before Sauron packages were installed. A comparison
scan of `python:3.11-slim-bookworm` produced 6 Critical, 20 High, and 70 Medium.
Downgrading to Debian 12 is therefore not recommended as a CVE-reduction step.

### 2. Raw occurrence counts overstate the number of distinct issues

Trivy reports one occurrence for every affected binary package. Examples from
this scan include:

- One `util-linux` CVE appearing against nine related packages.
- One GnuPG CVE appearing against seven related packages.
- Three curl CVEs appearing against `curl`, `libcurl3t64-gnutls`, and
  `libcurl4t64`, producing nine High occurrences.

The occurrences still require review, but prioritization should use both
unique CVEs and affected package/runtime reachability rather than the headline
count alone.

### 3. The fixable Python findings are mostly runtime build tooling

Path-backed findings were present in the system Python installation inherited
from the base image:

- System `pip 24.0` -- four Medium findings.
- System `setuptools 79.0.1` -- one Medium finding.
- Vendored `jaraco.context 5.3.0` -- one High finding.
- Vendored `wheel 0.45.1` -- one High finding.

The application runs from `/opt/venv` and does not perform package installation
at runtime. Current secured copies are already present in that virtual
environment, so the system bootstrap tooling should be removable from the
final stage after validation.

The report also listed vulnerable `msgpack 1.1.2` and `setuptools 70.3.0`
packages without a filesystem path while fixed copies were present. Trivy
warned that the third-party BuildKit SBOM could produce inaccurate package
attribution. These two entries must be confirmed with a Trivy-native SBOM or
direct final-filesystem scan before they are treated as genuine or suppressed.

### 4. The build is not dependency-reproducible

`requirements.lock.txt` is not consumed by the Dockerfile. The build installs
from `requirements.txt` plus minimum-version constraints, so the resulting
image can change without a repository change. For example, the checked-in lock
listed an older cryptography package while the reviewed image resolved a newer,
non-vulnerable release.

This behavior can pick up security fixes automatically, but it prevents exact
reproduction and makes scan-to-source reconciliation difficult.

### 5. Dockerfile misconfiguration findings

`trivy config Dockerfile` produced:

- **High DS-0002:** the container runs as `root` because no final `USER` is set.
- **Critical DS-0031 (two occurrences):** `HF_TOKEN` and
  `HUGGING_FACE_HUB_TOKEN` are accepted as build arguments. Secret values in
  build arguments can leak through build metadata or logs.

The trusted CA bundle is intentionally copied into the image and is not a
secret. Authentication tokens must use BuildKit secret mounts.

## Existing positive controls

The current build already provides several useful controls that should be
preserved:

- Multi-stage image construction.
- Debian slim runtime with `--no-install-recommends`.
- CPU-only Torch installation, avoiding unnecessary NVIDIA/CUDA packages.
- Security minimum-version constraints and offline model verification.
- Package cache cleanup.
- BuildKit provenance and SBOM attestations.
- Hugging Face, PDF, and tokenizer assets baked for air-gapped operation.

## Remediation work plan

### Phase 0 -- Establish a controlled baseline

- [ ] Store the enterprise Trivy JSON report as a restricted CI artifact.
- [ ] Rescan the immutable image digest, never only `latest`.
- [ ] Record Trivy version, vulnerability DB update time, image digest, target
      platform, and scan flags with every report.
- [ ] Generate a Trivy-native CycloneDX/SPDX SBOM from the final image.
- [ ] Compare the native SBOM with the BuildKit SBOM and resolve pathless
      `msgpack` and `setuptools` entries.
- [ ] Produce two summaries: the complete report and the subset with vendor
      fixes available.

Acceptance criteria:

- Every Critical and High occurrence maps to a package path/layer or is marked
  as a confirmed SBOM attribution issue.
- Enterprise and CI scanner differences are explained by scanner/DB versions,
  not by an unidentified image difference.

### Phase 1 -- Remove fixable Python findings

- [ ] Remove unused system `pip`, `setuptools`, and their vendored packages
      from the final runtime stage; retain secured tooling in `/opt/venv` only.
- [ ] Add an explicit `msgpack>=1.2.1` security floor.
- [ ] Set the cryptography floor to the reviewed secure major version and run
      authentication/CA compatibility tests.
- [ ] Generate a Linux/amd64 runtime lock file and make Docker consume it.
- [ ] Ensure CPU Torch wheels remain sourced from the approved CPU repository.
- [ ] Run `pip check`, `pip-audit`, Trivy, unit tests, and the air-gapped smoke
      test against the resulting image.

Acceptance criteria:

- Zero fixable Critical or High Python findings.
- No vulnerable superseded package metadata remains discoverable in the final
  filesystem or SBOM.
- The installed package inventory matches the checked-in runtime lock.

### Phase 2 -- Correct secret handling and runtime identity

- [ ] Replace `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` build arguments with
      BuildKit `RUN --mount=type=secret` handling.
- [ ] Update GitHub Actions and enterprise build instructions to pass the token
      as a BuildKit secret.
- [ ] Create a dedicated unprivileged `sauron` user/group in the final stage.
- [ ] Make model and tokenizer caches readable by that user and non-writable at
      runtime.
- [ ] Grant write access only to `/app/data` and any explicitly documented
      temporary directory.
- [ ] Document Run:AI/Rancher security context values: non-root UID/GID,
      `allowPrivilegeEscalation: false`, dropped capabilities, and read-only
      root filesystem where compatible.

Acceptance criteria:

- Trivy DS-0002 and DS-0031 findings are cleared.
- No secret value appears in image history, provenance, build log, or final
  environment.
- Startup, ingestion, LightRAG, and model-cache access work as the non-root user.

### Phase 3 -- Reduce native Debian package exposure

Perform these as separate measured experiments so document-processing
capabilities are not silently lost.

1. **Remove curl**
   - [ ] Replace the Docker health check with a small Python standard-library
         request.
   - [ ] Remove direct `curl` installation and run `apt autoremove` safely.
   - [ ] Measure whether `libcurl3`/`libcurl4` remain as transitive dependencies.

2. **Use headless OpenCV**
   - [ ] Test `opencv-python-headless` in place of GUI OpenCV.
   - [ ] Attempt removal of `libgl1` and `libglib2.0-0`.
   - [ ] Validate scanned-PDF layout detection, OCR, tables, and figure review.

3. **Review Poppler and other native utilities**
   - [ ] Trace which Sauron code paths still require `poppler-utils` versus
         PDFium/pdfplumber.
   - [ ] Test removal only if scanned and digital PDF behavior remains intact.
   - [ ] Review GnuPG and general utility packages inherited from the Python
         base for safe removal.

4. **Apply current Debian security packages**
   - [ ] Rebuild frequently from the current approved base digest/update.
   - [ ] Confirm `apt-get update` and installation occur in one layer.
   - [ ] Do not claim an OS finding is fixed unless Debian supplies and the
         final image installs a fixed version.

Acceptance criteria:

- The full Critical/High count decreases without feature regression.
- PDF, Word, PowerPoint, spreadsheet, OCR, image analysis, KG, tokenizer, and
  offline inference smoke tests pass.

### Phase 4 -- Add CI scanning and enforcement

Modify `.github/workflows/docker-publish.yml` to implement:

- [ ] A local `linux/amd64` image build for pull requests.
- [ ] A complete Trivy JSON and SARIF report with `exit-code: 0` for visibility.
- [ ] A second gate for fixed Critical/High vulnerabilities using
      `--ignore-unfixed --severity CRITICAL,HIGH --exit-code 1`.
- [ ] Report artifacts with an appropriate enterprise retention period.
- [ ] SARIF upload to GitHub code scanning when repository settings permit.
- [ ] Push only after the pre-publish gate succeeds.
- [ ] A post-publish scan by immutable digest.
- [ ] A scheduled scan so newly disclosed CVEs are detected without a source
      commit.
- [ ] A scheduled base-digest and dependency-lock update process.

For the air-gapped enterprise pipeline:

- [ ] Mirror the Trivy vulnerability database into an approved internal OCI
      registry.
- [ ] Configure `TRIVY_DB_REPOSITORY`/scanner cache according to enterprise
      policy.
- [ ] Record the internal DB snapshot identifier with each report.

Acceptance criteria:

- No image is published with a newly introduced fixable Critical/High finding.
- Full unfixed findings remain visible and trendable rather than being hidden by
  the build gate.

### Phase 5 -- Handle currently unfixed OS Criticals

There are two policy paths:

1. **Risk-managed Debian path**
   - [ ] Assess reachability for each exact package/CVE/image digest.
   - [ ] Create a VEX or enterprise risk-acceptance record only when evidence
         shows the vulnerable code is not executable through Sauron.
   - [ ] Include owner, rationale, compensating controls, expiration date, and
         rescan trigger.
   - [ ] Never use a blanket severity or package ignore.

2. **Zero-Critical runtime path**
   - [ ] Prototype a Wolfi/Chainguard or distroless runtime.
   - [ ] Account for Bash entrypoint behavior, Tesseract, Poppler/native
         libraries, CA injection, offline model caches, and first-start seeding.
   - [ ] Compare CVEs, image size, support lifecycle, enterprise mirror
         availability, and operational complexity.

Acceptance criteria:

- If policy permits unfixed findings, every remaining Critical has a current,
  digest-specific approved record.
- If policy requires an absolute zero-Critical scan, the alternative runtime
  must meet that gate and pass the complete functional test matrix.

## Suggested implementation order

1. Phase 0: baseline and native SBOM verification.
2. Phase 1: remove fixable Python/tooling findings.
3. Phase 2: secret mounts and non-root execution.
4. Phase 4: CI visibility and fixable-vulnerability gate.
5. Phase 3: native dependency reduction experiments.
6. Phase 5: VEX/risk acceptance or alternative runtime decision.

This order produces a trustworthy scanner baseline and removes deterministic
findings before taking higher-risk actions that could affect OCR and document
processing.

## Reproduction commands

Use the enterprise-approved Trivy binary and DB mirror. Replace the digest when
reviewing a newer build.

```bash
# Full report -- do not suppress unfixed findings
trivy image \
  --scanners vuln \
  --format json \
  --output trivy-full.json \
  ghcr.io/mulkeym/sauron@sha256:afbe7c810eb2c3d120f10d754c999daa83bbbbd9fa5d58bf1cd03cf088d464fd

# Actionable fixed-version gate
trivy image \
  --scanners vuln \
  --ignore-unfixed \
  --severity CRITICAL,HIGH \
  --exit-code 1 \
  ghcr.io/mulkeym/sauron@sha256:afbe7c810eb2c3d120f10d754c999daa83bbbbd9fa5d58bf1cd03cf088d464fd

# Dockerfile hardening review
trivy config Dockerfile

# Trivy-native SBOM for comparison with BuildKit output
trivy image \
  --format cyclonedx \
  --output sauron.cdx.json \
  ghcr.io/mulkeym/sauron@sha256:afbe7c810eb2c3d120f10d754c999daa83bbbbd9fa5d58bf1cd03cf088d464fd
```

## Final completion checklist

- [ ] Full report retained and attributable to an immutable digest.
- [ ] Zero fixable Critical/High language findings.
- [ ] No build secrets in arguments, history, logs, or environment.
- [ ] Runtime executes as non-root.
- [ ] CI gate and scheduled rescans enabled.
- [ ] Remaining unfixed Criticals have approved, expiring evidence or are
      eliminated by a different runtime.
- [ ] Air-gapped functional tests pass for all supported document types and KG.
- [ ] Release checklist records the image digest, SBOM, scan report, and
      approved exceptions.
