# Release verification and rollout

The release includes technical-source retrieval and graph-guided follow-up,
readable MCP citations, diagram discovery, Visio text-layout and EMF rendering
repairs, original-file retention/downloads, warm CPU embedding workers, and
admin controls for embedding performance and final-answer reasoning/temperature.

## Container publication gate

The Docker workflow builds linux/amd64 once into an OCI layout. Every exported
layer, including attestation layers, must be **less than 1,000,000,000 bytes**.
This is the compressed downloadable layer limit, not a limit on total image size.
Python dependencies and offline model caches remain split into 16 layers each,
with an 850,000,000-byte uncompressed-file budget per partition.

The workflow loads the exact artifact, checks native tools and offline startup,
and runs automated regressions against its packaged source with isolated data.
PR builds run these gates without publishing. Successful default-branch/tag
builds publish the same OCI artifact and its attestations. A failed check leaves
release tags, including latest, unchanged. Layer measurements and startup logs
are attached to the Actions run. Use the recorded SHA tag/digest for rollout.

## Local verification

- Existing PDF and Visio import/visual acceptance tests are accepted; no repeat
  manual import campaign is required for these publication changes.
- Full automated suite: 1,270 passed, one optional browser test skipped.
- Container build helper tests after adding the OCI layout gate: 12 passed.
- GitHub Actions syntax validated with actionlint.
- OpenWebUI companion backend compatibility: 42 passed; optional cross-service
  browser test skipped (already exercised during integration work).
- Packaged-source smoke image starts offline, seeds a fresh private volume,
  serves admin login, rejects anonymous API access, and accepts the test API key.
  This local smoke image reuses the arm64 development dependency layers; the
  clean production amd64 build is verified separately by GitHub Actions.

The regression review corrected two tests that unintentionally depended on a
local metadata database, and updated endpoint enumeration for FastAPI's lazy
included routers. Authentication expectations were not relaxed.

## Rollout order

1. Review the release PR and its native amd64 image checks.
2. Merge after the checks pass; verify the publishing run and retain its digest.
3. Release the paired OpenWebUI extension before enabling native citation cards
   or original-document links in that client. Its code remains in the separate
   OpenWebUI repository; a Sauron-only release does not deploy that extension.
4. Back up metadata, settings, figures and retained originals together, record
   the previous image digest, then use the verified SHA/digest on Proxmox when
   deployment is requested. Existing persistent mounts must remain in place.

Original retention is opt-in and only captures new ingestions until exact
originals are backfilled. Original downloads require the dedicated shared secret
and matching OpenWebUI configuration. See original-downloads.md and citations.md.
Technical section chunking, procedure retrieval and troubleshooting retrieval
remain off by default; revision selection is on. Saved admin settings take
precedence. No settings change or document re-ingestion is part of publication.
