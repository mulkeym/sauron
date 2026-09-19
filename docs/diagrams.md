# Stored source diagrams

Sauron retains PNGs extracted from newly ingested PDFs, Word documents, PowerPoint pictures, Excel images and rendered Visio `.vsdx` pages. See [Visio ingestion](visio.md) for conversion behavior and fidelity limits. It stores source images and page renders, including network topologies and Venn diagrams. It does not generate replacement illustrations.

Images live under `/app/data/figures` in the existing persistent data volume. No new mount, service or model is required. Visio rendering adds the native tools described in [Visio ingestion](visio.md). Figures retain their source document's access restrictions. PNG bytes are absent from vector embeddings, query caches, extraction JSON, and answer text.

## Use

- Ask a diagram question through `tool_ask` for a grounded explanation with images from cited figure evidence.
- Use `tool_search_diagrams(query, top_k=5, doc_id="", kind="")` for ranked diagram candidates. It returns metadata, not binary images. Verify the candidate's source/site/version before applying it.
- Use `tool_get_diagram(doc_id, figure_id, variant="preview")` to retrieve a selected source image without answer synthesis. `full` requests the original normalized PNG when retained within limits.
- Native MCP responses include text/structured metadata and separate base64 `ImageContent` blocks. Do not ask the language model to copy base64 into Markdown.
- REST search: `GET /api/v1/diagrams/search?query=branch%20topology`.
- REST PNG: `GET /api/v1/documents/{doc_id}/figures/{figure_id}/content?variant=preview`. Requires the usual API key and user Bearer token. It is not a public Markdown image URL; backend clients must fetch and present it through their own authorized attachment handling.
- Query and async-query responses include an `images` array with source metadata and authenticated content URLs. The OpenAI-compatible endpoint provides `sauron_images` as an extension; generic OpenAI clients may ignore it. Native MCP is the intended OpenWebUI image integration.

OpenWebUI must support native MCP image results. Its attachment rendering/storage is separate from whether a vision model can inspect the image. Sauron permission changes stop subsequent retrieval; copies already delivered into a client's chat follow that client's retention and sharing controls.

## Admin portal

**Diagrams** provides permission-scoped search, previews, document image counts, analysis counts, storage usage and image backfill. **Settings → Advanced** exposes all image settings. **Answer Profiles** lets each profile inherit the system image policy or choose auto, requested or off, with its own image count bounded by the system maximum. Playground and profile previews display cited PNGs.

Defaults:

| Setting | Default |
|---|---|
| Store figures from future ingestions | On |
| Retained figure candidates per document | 100 |
| Total PNG storage per document | 100 MiB |
| Maximum decoded image pixels | 16 million |
| Full PNG variant | At most 12 MiB |
| Preview | At most 2,000 pixels on its longest edge and 3 MiB |
| Vision analysis | Existing 20-distinct-image budget |
| Answer images | Auto; up to 2 |
| Total native MCP base64 image data | At most 8 MiB |

Storage and visual analysis are independent: a vision timeout can leave an available image searchable by caption, OCR and neighboring text. Full variants exceeding the byte limit are omitted; previews may be reduced to fit. Storage/analysis omissions appear as ingestion warnings and in the inventory. Changing ingestion limits does not rebuild existing assets. Disabling retention affects future ingestion, not existing figures.

## Existing documents

Earlier Sauron versions discarded image bytes. Open **Diagrams**, choose a document eligible for backfill, and supply the matching original source. Its content hash must match the ingested document. Backfill preserves document identity, text, tables and ACLs and refreshes dedicated figure search records. It only accepts documents without stored images. Documents lacking an original content hash, changed sources, and replacing existing images require re-ingestion through the source workflow.

The source document remains externally managed. Sauron's PNGs are derived retrieval assets; the source upload is still temporary. The storage module isolates binary storage behind asset keys for future remote-storage support. A remote backend is not included in this release.

## Coverage and recovery

Embedded image extraction is supported. PDF pages containing substantial vector paths can also be rendered, including text-heavy pages with vector-drawn Venn diagrams. These are labeled page renders and may include surrounding prose/tables; automatic tight diagram cropping is not implemented. Native Office shapes and SmartArt are not fully rendered by picture extraction.

Extraction and image decoding stay in the disposable worker in the same container. Parent/child handoff uses staged files and small manifests. Pending images are inaccessible until their owning document exists. Failed jobs remove their assets; startup removes abandoned staging and assets without a live document. Deleting a document removes its figure metadata and PNGs. Backups include published images and exclude extraction/staging directories. Create backups while ingestion/backfill is idle.
