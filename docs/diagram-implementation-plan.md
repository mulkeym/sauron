# Sauron: stored diagrams and image retrieval

Implementation proposal — 19 September 2026. Based on local Sauron master at `dd57b38` and the preceding inspection of the deployed OpenWebUI image. This is the original design proposal; see [diagrams.md](diagrams.md) for the implemented behavior and current limitations. Deployment is separate.

## Intended behavior

A user asks “Show me the branch SD-WAN topology” or “Explain the failover design and include its diagram.” Sauron searches permitted documentation, identifies the relevant figure, and returns its source image with a caption, document reference, and page or slide location. An explanation can accompany the image. An image-only request does not require successful answer synthesis.

Network diagrams, Venn diagrams, process diagrams, charts, and other relevant illustrations can all be retained as PNGs. These are extracted source images or faithful page/region renders. Sauron does not recreate diagrams with an image-generation model. A Venn diagram drawn using PDF vector shapes requires rendering; saving embedded raster images alone will not cover it.

Keep one Sauron container and the existing disposable extraction worker. Reuse PDFium, Pillow, OCR, vision descriptions, and text embeddings. The initial implementation needs no new large model or document-parsing framework.

## Current foundation and gaps

- `src/ingestion/figure_extract.py` collects PDF and Office images and produces descriptions with placement metadata. Image bytes exist temporarily in `ImageRegion` but are absent from the returned `FigureRecord`.
- `prepared.py` serializes extraction results as JSON. `isolation.py` deletes the extraction task directory in its `finally` block before its caller can consume any files there.
- Both `pipeline.py` and `queue.py` index dedicated figure chunks. Ordinary text chunks also contain figure descriptions, so those matches can lose a direct figure association.
- PDF extraction collects embedded raster objects and renders pages with fewer than 40 digital text characters by default. Vector diagrams with many labels or surrounding prose can be missed. Office picture extraction does not fully render native shapes, SmartArt, or complete slide compositions.
- Figure analysis defaults to 20 distinct images per document. The collector currently accumulates image bytes before that processing budget is applied. Storage and analysis need separate limits applied early.
- Figure records currently depend on successful descriptive output. Retention should survive OCR or vision failure when the image has a usable caption or other source context.
- Citations preserve figure IDs, but answer/API models have no image attachment collection. The low-level search tool drops some existing figure metadata.
- Query-cache validity includes catalog, index, settings, and answer-profile revisions; it needs an explicit figure revision for asset-only changes.
- Source uploads are temporary and removed after processing. Existing ingestions cannot yield their original images without re-accessing the source.

## 1. Figure metadata and binary storage

Add a figure occurrence table in the metadata database, keyed by `(doc_id, figure_id)`. Record:

- Opaque asset key, full SHA-256 of the stored PNG, MIME type, encoded byte size, width, and height.
- Page/slide, bounding box, coordinate convention, section, caption, alt text, and source locator where known. Use one-based page/slide numbers at API boundaries.
- Figure kind, searchable description, OCR text, neighboring source text, analysis status, extraction method/version, and asset status.
- Owning source content hash and a document-level figure revision, so refreshes cannot silently associate a figure with a different document version.

Store files under `/app/data/figures/{doc_id}/{sha256}.png`. Repeated occurrences in one document can share bytes while retaining their individual IDs, captions, and placements. Avoid cross-document deduplication initially; this keeps ACLs and cleanup simple.

Use a small `FigureStore` interface for stage, publish, open, and delete operations. The initial backend uses the existing persistent `/app/data` mount. Asset keys, rather than absolute filesystem paths, are stored in the database. A future remote object store can implement the same interface; building that backend is deferred.

The remote system remains the source-document authority. Sauron's PNGs are derived retrieval assets, like its existing text index. This does not require a local document-management workflow.

## 2. Extraction and safe publication

The child writes normalized PNGs to a job-owned staging directory and includes a small manifest in its JSON result. Do not put base64 into extraction JSON, vector rows, or model prompts.

Introduce an explicit extraction artifact lifetime: the parent validates and transfers staged files into an ingestion-owned staging area before the extraction directory is removed. Give both synchronous and queued ingestion a shared ownership/cleanup helper. Files must remain usable through indexing and must be removed on cancellation or failure.

Validation covers relative paths, regular files only, no symlink/path escape, byte counts, PNG signature, and hash matching. Image decoding, normalization, resizing, and pixel-limit checks stay in the disposable child; the API handles bounded file I/O and metadata.

Publish assets atomically on the same filesystem, and make them retrievable only once their owning document and figure metadata are committed and ready. SQLite, LanceDB, and files do not form one transaction: track staging/publication explicitly, roll back a failed job's writes, and reconcile interrupted jobs on startup. Never expose files solely because they exist on disk.

Process images incrementally and release pixel buffers after writing/analysis. Apply count, pixel, per-image byte, per-document byte, and timeout limits during collection, including page rendering. Keep current memory supervision and API headroom.

Retention and vision analysis are separate. Save acceptable figures even if vision times out; index available caption, alt text, section, neighboring text, and OCR. Mark missing analysis rather than inventing a description. Report omissions and limits in ingestion status. A broken optional figure should produce a partial-extraction warning while ordinary text ingestion can complete; unsafe or inconsistent job artifacts must fail publication.

## 3. Search and direct retrieval

Extend vector search with a `content_type=figure` filter applied before top-k selection. Use the current hybrid text/vector retrieval and reranker on descriptions, captions, OCR labels, and section context. Preserve source-derived text separately from generated interpretation.

Add:

- `tool_search_diagrams(query, top_k=5, doc_id=None, kind=None)`: returns compact figure matches and source metadata, with no binary payload.
- `tool_get_diagram(doc_id, figure_id, variant="preview")`: returns the permitted PNG plus its caption/provenance.
- REST equivalents for search and authenticated figure content, for other clients.

Search results include figure ID, document ID, caption, kind, page/slide, source locator, description, dimensions, availability, and relevance. General document-search results also retain figure metadata and advertise whether an image is available.

For topology, Venn, or diagram requests, perform a dedicated figure search in addition to ordinary passage retrieval. Match site/platform/version constraints where the source supplies them. Preserve figure references through retrieval strategies and summary/reduction steps; add direct figure evidence to synthesis when needed. Do not rely on the model reproducing an ID from prose or finding figures only among text top-k results.

Filter using authoritative current document permissions and dataset scope at search time, then check again when delivering bytes. Reuse the current authenticated API-key plus forwarded-user/group context for MCP. Do not accept permissions as ordinary tool arguments. Use consistent not-found responses for inaccessible asset IDs.

## 4. Answers and client delivery

Extend internal responses with an `images` collection of references and metadata. Propagate it through normal queries, streamed queries, async job results, cache results, MCP, and admin Playground. Keep binary delivery at the transport boundary.

Recommended behavior:

- Default `auto`: include clearly relevant diagrams for visual questions and diagrams explicitly cited by the answer.
- `requested`: attach only when the user asks for images.
- `off`: return text and normal citations.

For generated explanations, attachments must refer to retrieved, permitted figure evidence; validate selected IDs just as citation IDs are validated. If clarification is needed for a site/version-specific topology, do not present a candidate as the confirmed applicable topology. Direct diagram search can show labeled candidates without generating unsupported explanatory claims.

Return MCP `TextContent` containing answer/search metadata plus actual `ImageContent` blocks with base64 PNG data. Preserve the existing structured answer fields for other clients. Do not nest raw image blocks inside a JSON dictionary that FastMCP would serialize as ordinary text.

Base64 is used only when sending the image; keep it out of answer Markdown, logs, audit records, query caches, and text-model context. Add source labels alongside images and explicit ordering/figure IDs in metadata.

For REST clients, return authenticated content endpoints. Ordinary Markdown image requests do not automatically carry the calling backend's API key and forwarded identity. The client must fetch through its authenticated backend and expose a user-authorized attachment, or use an authenticated same-origin viewer. Never place the shared API key in an image URL. The admin Playground can use its existing protected session to load previews.

OpenWebUI source inspection found handling for native MCP image blocks and protected file storage. The first development milestone must verify the actual deployed chat path: visibility in tool results, placement in the final assistant response, persistence after reload, and no raw base64 in model-facing text. Passing an image to a vision model and showing it to the user are separate capabilities. If final-message promotion is missing, scope a small OpenWebUI compatibility change separately.

A diagram already copied into OpenWebUI belongs to OpenWebUI's chat/file lifecycle. Revoking Sauron access blocks future retrieval but cannot retroactively remove that delivered copy. Document this boundary and verify OpenWebUI's attachment access behavior.

## 5. Extraction coverage

First deliver retention and retrieval for images the existing collectors already find, including embedded Venn diagrams. Then improve vector coverage:

1. Identify substantial PDF drawing regions using paths/shapes, captions, text layout, and grouping; avoid treating ordinary table borders as topology figures.
2. Render candidate regions using existing PDFium, retaining text labels, connecting lines, and legends. Provide a labeled page-render fallback where crop boundaries are uncertain.
3. Index these renders using the same description/OCR and asset pipeline. Include text-heavy pages in evaluation; the existing sparse-page rule is insufficient.
4. Treat complete Office shapes/SmartArt/slide rendering as a separate extension. Evaluate conversion/rendering dependencies before adding them to the container. Do not claim support for these through picture extraction alone.

Keep original source geometry/provenance so users can distinguish an embedded image, cropped page region, or entire rendered page.

## 6. Admin controls and lifecycle

Expose every new setting in the admin portal, with descriptions, validated bounds, and restart indicators where required. Add clear figure controls in Document Processing and image behavior in Answers/Answer Profiles; the advanced catalog must include them too.

Suggested starting values, to be tuned against legibility and memory tests:

| Setting | Proposed initial value |
|---|---|
| Retain extracted figures | Enabled for new ingestions after release |
| Automatic answer attachments | Auto, maximum 2 |
| Search results without binaries | Default 5, bounded maximum |
| Preview dimensions | Longest edge 2,000 px, no upscaling |
| Preview bytes | Maximum 3 MiB per PNG |
| Total inline MCP image data | Maximum 8 MiB after base64 encoding |
| Stored full-resolution PNG | Preserve within separate pixel/byte limits |
| Figures retained per document | Initial cap 100 distinct assets |
| Vision analysis budget | Keep existing default 20 distinct figures |
| Figure bytes per document | Initial cap 100 MiB |

Define explicit decoded-pixel and full-resolution-byte caps during implementation using the existing memory budget. Counts alone do not bound memory. Separate preview and full-resolution variants so shrinking for chat does not destroy the only readable copy. If a limit prevents an acceptable preview, return availability metadata and a clear omission reason; do not claim an omitted image was attached.

Add Playground image previews and extraction/storage counts: detected, stored, analyzed, omitted, missing assets, and bytes used. Settings describe whether changes affect future ingestion, future retrieval, or require reprocessing. Changing storage paths requires a controlled migration/restart, not an immediate live pointer change.

Integrate cleanup into API deletion, admin deletion, failed sync/async jobs, source replacement, and maintenance. Purge document-owned assets once and avoid deleting active staging data. Startup reconciliation must distinguish interrupted staging from valid assets and refuse broad cleanup when metadata cannot be trusted.

Include published PNGs and matching metadata in backups; exclude temporary staging. Make backup creation consistent with publication, and verify restore retains resolvable image references. Old backups without figures must remain supported.

Cache image references and source/figure revisions, never base64 or expiring links. Revalidate access and availability for cached answers. Asset-only refreshes invalidate affected results; missing files yield bounded warnings and usable text where possible.

## 7. Existing documents and staged rollout

Add schema fields with compatible defaults. Existing documents initially show no stored images. Re-access original documents through the ingestion source/connector or an explicit source submission; extracted text cannot reconstruct the missing pixels.

Provide a controlled figure-backfill operation keyed by existing document ID. Verify the supplied source hash matches the ingested source. Backfill retains document identity and ACLs, replaces only figure assets/index records, updates the figure revision, and invalidates affected caches. For documents without a stored source hash, require an explicit operator association. If the source changed, treat it as source replacement rather than silently reusing old figure identities. This avoids the current duplicate-ingestion rejection and delete-first workflow for image backfill.

Do not automatically fetch arbitrary source URLs on a user query. Reprocessing uses configured source mechanisms and the same isolated extraction pipeline.

Recommended implementation sequence:

1. **Transport proof:** use a synthetic, non-sensitive diagram to validate native MCP text plus image delivery in the deployed OpenWebUI flow. This can be a disposable development endpoint/test harness; it need not alter production configuration.
2. **Storage and lifecycle:** metadata migration, bounded PNG variants, worker artifact handoff, publication, cleanup, and backup/restore coverage in both ingestion paths.
3. **Search and retrieval:** figure filtering, metadata-rich search, authenticated image delivery, direct MCP tools, and Playground previews.
4. **Answer integration:** supplementary figure retrieval, validated attachments, response-model propagation, profiles/admin settings, and cache revisions.
5. **Coverage and migration:** PDF vector-region rendering, source-verified figure backfill, then representative on-prem evaluation before enabling wider ingestion.

Milestones 1–4 form a useful first release for embedded diagrams. Milestone 5 is required before promising broad coverage of vector-drawn PDF topologies or Venn diagrams.

## Acceptance tests

- An embedded network or Venn diagram is ingested, survives container restart, is searchable by topic and labels, and can be retrieved as a valid PNG with correct page/slide provenance.
- Duplicate bytes retain multiple placements without duplicate storage within one document.
- OCR/vision failure retains a usable image and source context with an explicit analysis status.
- A text-heavy PDF containing a vector diagram is found after the vector-coverage milestone; labels, connectors, and legends remain visible.
- “Show me the diagram” works without synthesis; an explanation can include a permitted cited image. No-match and ambiguous-site/version cases behave clearly.
- Cross-group search and guessed asset IDs fail; ACL changes and deletion block fresh retrieval and cached-result delivery.
- Worker crash, timeout, cancellation, disk-full conditions, malformed images, and interrupted publication leave the API usable and no retrievable partial assets.
- Search filters apply before top-k so numerous text matches cannot hide the relevant figure.
- Preview and full-resolution limits are enforced without putting image data in IPC JSON or text-model input.
- Sync, async, streaming, cache-hit, MCP, REST, and Playground paths preserve image references and useful text fallback.
- OpenWebUI shows the returned image in the tested chat path after reload; attachment access and client-side persistence are verified separately from model vision support.
- Deletion, backfill, backup/restore, old-schema upgrade, and old backups preserve or remove the correct references and files.

## Main implementation locations

New figure storage/retrieval modules and metadata model; updates to `figure_extract.py`, `prepared.py`, `isolation.py`, `extraction_worker.py`, `prepared_index.py`, `pipeline.py`, and `queue.py`; vector filtering and query scope/cache revisions; response types and synthesis/strategy reference handling; MCP tool registration/results and authenticated REST routes; admin settings, profiles, Playground, deletion, and backup/restore paths. Consolidate shared figure-index/publication code so sync and queued ingestion use the same implementation.
