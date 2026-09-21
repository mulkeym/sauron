# Authenticated original-document downloads

`tool_get_original_document(doc_id, revision)` returns document details and an OpenWebUI URL. Pass the citation's `source_revision` (the SHA-256 of the indexed bytes) as `revision`. Human revision labels such as `edition.revision = "Rev 2.0"` are not byte identifiers. If an older citation lacks `source_revision`, use `tool_lookup_document` on that exact `doc_id`; a deleted ID fails instead of resolving to a newer filename or family.

The URL contains only document and revision identifiers. The browser sends its existing OpenWebUI session cookie. OpenWebUI verifies that session and reads current group membership from its database. It sends its Sauron application key plus a signed identity assertion with a stable user ID, issuer, audience, operation, document ID, revision and a 60-second lifetime. The assertion stays between the backends. No token exchange, API key, reusable bearer link, or original bytes appear in model context.

Sauron validates the application key and assertion independently, then reads the current document ACL and dataset state before every GET/HEAD/range request. Missing/inactive datasets deny delivery. `default_acl_groups` remains an ingestion default; it is not a dataset ACL. An empty group set or the special `ALL` group never grants download access. Ordinary OpenWebUI administrators also require a matching document group. Group names must match Sauron's ACL names.

## Configure both services

Sauron (environment variables, or corresponding lower-case settings):

```dotenv
SOURCE_ORIGINALS_DIR=/app/data/originals
SOURCE_ORIGINALS_MAX_MB=2048
SOURCE_DOWNLOAD_WEBUI_URL=https://chat.example.org
SOURCE_DOWNLOAD_JWT_SECRET=<dedicated random secret of at least 32 characters>
```

OpenWebUI: use **Admin Settings → Integrations → Sauron original downloads** to configure the backend URL, API key and signing secret. Blank credential inputs preserve saved values. The following server environment defaults are also supported:

```dotenv
SAURON_DOWNLOAD_BASE_URL=https://sauron.internal.example.org
SAURON_DOWNLOAD_API_KEY=<dedicated active Sauron application key>
SAURON_DOWNLOAD_JWT_SECRET=<same dedicated signing secret>
```

`SOURCE_DOWNLOAD_WEBUI_URL` can include OpenWebUI's reverse-proxy path prefix. Do not include a query, fragment, username or password. Use HTTPS in production. Loopback HTTP is accepted for local tests. To use a deliberately isolated internal HTTP network, an administrator must explicitly set `SAURON_DOWNLOAD_ALLOW_HTTP=true`; protect that channel because it carries the API key and signed identity. HTTPS certificate verification stays enabled; redirects are refused.

The existing MCP connection and its user/group headers continue to serve retrieval. The original-file endpoint does **not** accept the legacy unsigned identity headers or a generic MCP JWT. Its separate purpose-scoped assertion prevents a retrieval identity token from being reused for another operation.

Mount `SOURCE_ORIGINALS_DIR` as private persistent storage available to all Sauron replicas. It must not be a static/public directory, and only the Sauron service and trusted administrators should write it. OpenWebUI stores no permanent copy. Configure reverse proxies to disable shared caching and response buffering for these endpoints, and to avoid logging credential headers. Application audit events include subject, document ID, revision, method and outcome, never credential values.

## Retention and existing documents

Retention is opt-in: without `SOURCE_ORIGINALS_DIR`, new ingestion continues and download requests report originals unavailable. When configured, both direct and queued ingestion copy the original before removing the temporary upload. Retention verifies SHA-256, enforces the size bound, and publishes atomically without overwriting an existing revision. A retention failure fails ingestion rather than advertising an available original.

The implemented provider is a private local/shared filesystem. Files are addressed by validated `doc_id` and SHA-256, not by `source_url`, filenames, arbitrary paths or URLs supplied by a browser/model. Supported retained types include PDF, Word, Visio, PowerPoint and Excel. Converted web-page Markdown is not advertised as an original website or document. Existing crawled binary downloads can be retained on re-ingestion; no runtime connector fetch is performed. Box, SharePoint or other external providers are not implemented.

Existing Sauron uploads were deleted after ingestion. Recover the exact original from its owner, backups or source system and run:

```sh
python -m scripts.backfill_original --doc-id DOCUMENT_ID --file /private/path/original.docx
```

The CLI reads the authoritative catalog and refuses bytes that do not match `content_hash`. It does not reindex or change document permissions. Empty/legacy hashes require verified re-ingestion; do not attach an arbitrary replacement to an old citation. Byte-identical originals are required. Deleting a catalog document also deletes its retained directory. Back up catalog and originals together. Interrupted ingestion can leave inaccessible orphan bytes that require administrator cleanup; there is no automatic retention expiry or external-object garbage collector.

## Delivery behavior and limits

PDFs with a PDF signature use inline viewing; Word and Visio use attachments. Other supported Office types use conservative attachment MIME types. Filenames are sanitized and encoded. All deliveries have `private, no-store`, `nosniff`, a sandbox policy, and no-referrer headers. Unauthorized and unknown documents share a 404 response. An authorized user requesting a mismatched revision receives 409; unavailable/corrupt original bytes produce 410. Single byte ranges, suffix ranges, HEAD and If-Range are supported; multiple ranges return 416. Conditional 304 caching is deliberately not used.

Files are verified by streaming their entire SHA-256 before delivery, then streamed in 256 KiB chunks from the same open descriptor. RAM use is bounded, but each range request pays a full-file integrity scan. Very large PDFs may therefore need a future trusted immutable object-store provider for lower latency. Files must remain immutable after publication. Revocation applies to later requests, including later ranges; an authorized response already in progress or a saved local copy cannot be revoked. Browser/backend session revocation follows OpenWebUI's existing session implementation (including Redis-backed logout revocation where configured).

## Integration with retrieval changes

This branch starts at Sauron `f11eada835efcccb32235917ef2fb1062c8482b6`. The parallel retrieval task adds citation `source_revision` and edition provenance. Preserve both sets of additions in `src/mcp/tools_low.py`, `src/mcp/server.py`, `src/config.py`, and the retention hooks just before `add_document` in the pipeline/queue. Each changed-byte original must retain its distinct `doc_id`; do not replace an old document's content under the same ID. Download service behavior is independent of retrieval strategy and edition selection.

The matching OpenWebUI branch starts at `01f4282f1`. Its changes are a new backend router and one router registration; no changes to the shared integration-settings work are needed. A normal model-produced Markdown link is sufficient, so no new browser-side credential or blob-fetch code is required.

## Verification

```sh
python -m pytest tests/test_sources tests/test_mcp/test_auth.py tests/test_mcp/test_tools_low.py tests/test_mcp/test_server.py tests/test_auth/test_http.py tests/test_ingestion/test_pipeline.py tests/test_ingestion/test_queue_cleanup.py --asyncio-mode=auto
```

The matching OpenWebUI repository contains `backend/tests/test_sauron_downloads.py`, using its real session validator and user/group models with an isolated database. Optional cross-repository/browser verification is described in that repository's download documentation.
