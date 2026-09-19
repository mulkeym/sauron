# Visio ingestion

Sauron accepts modern `.vsdx` documents and renders whole pages with **libvisio + librsvg**. Microsoft Visio, exported companion images, a second service, and a commercial SDK are not required. Legacy `.vsd`, `.vdx`, and macro-enabled `.vsdm` are explicitly rejected in this release.

## What is retained

The worker reads the VSDX package's native page names, shape text, groups, properties, and connector attachments. It resolves available master text and cached string fields; it does not evaluate ShapeSheet formulas or refresh external data. Attachments establish drawing connections, not traffic direction, routing protocols, or failover intent. Original shape/page IDs remain in the source evidence.

For graphics, a temporary source copy materializes inherited text, saved string field values, and text formatting that libvisio otherwise sometimes omits. Geometry, connectors, pictures, master definitions, and backgrounds remain source-defined. The original source is unchanged. Libvisio outputs foreground pages followed by backgrounds; Sauron maps these back to their source page names, IDs, and package-order page numbers before attaching citations. Backgrounds are composited behind foreground diagrams and are also retained as separate pages.

Librsvg produces a white-canvas full-page PNG, normally up to 4096 pixels on the longest edge, subject to the existing pixel cap. The existing figure store retains full/preview variants within configured budgets. Large page descriptions are split into bounded search chunks that all retain the same page/figure citation, so labels near the end remain searchable. Page records enter the same figure indexing, permission checks, image retrieval, backfill, and native MCP attachment paths as PDF/Office figures. No base64 is inserted into embeddings or extraction JSON.

## Fidelity and warnings

These are converted previews, not certified pixel-identical Visio exports. Fonts, dynamic fields, data graphics, arrow styling, OLE/metafile objects, and unusual shapes may differ. Simple checks flag fewer embedded raster images or missing source text in the SVG; these checks cannot prove complete visual fidelity. Conversion warnings travel with figure references and appear in the admin preview. A page-count mismatch withholds images to prevent incorrect citations. A failed page does not discard successfully converted pages or native source text.

Testing against a public four-page network/rack sample found that materializing inherited fields restores many otherwise absent device names and IP addresses. Some complex callout labels still clip or misalign, and source-text coverage warnings remain. The source properties remain searchable. Another public, programmatically generated network VSDX could be parsed for text but was rejected by libvisio; ingestion correctly reported the missing PNGs. Neither result establishes universal compatibility.

## Configuration

All settings are available in **Admin → Settings → All Settings**:

| Setting | Default |
|---|---|
| `visio_enabled` | true |
| `visio_max_pages` | 100 |
| `visio_max_shapes` | 20,000 per page |
| `visio_max_unpacked_mb` | 256 MiB |
| `visio_converter_max_mb` | 64 MiB per converter output |
| `visio_timeout_seconds` | 120 seconds per native process |
| `visio_render_max_edge` | 4096 pixels |

Overall extraction memory/time limits and figure count/pixel/storage limits also apply. Both `figure_extraction_enabled` and `figure_store_enabled` must be on to retain PNGs. Disabling rendering keeps the structured source text and emits a warning.

The image adds Debian `libvisio-tools`, `librsvg2-bin`, `fonts-dejavu-core`, and `fonts-liberation`; `lxml` was already present and is now an explicit requirement. The measured incremental installed-package size on Debian trixie ARM64, after Sauron's existing system dependencies, was 28,008 KiB (about 27.4 MiB). This is not a compressed OCI layer-size measurement; architecture/package updates can change it. The existing layer-size build check still applies.

## Inspect without indexing

With Sauron's Python environment and native tools installed:

```sh
python scripts/inspect_visio.py diagram.vsdx --output /tmp/visio-inspection
```

Use an empty output directory. The utility runs the disposable worker, writes page PNGs, `source-text.txt`, and `report.json`, and does not add documents to the database. Exit status 2 means no pages were rendered; inspect the report for the cause. The source is never uploaded externally by this utility.

## Validation

Tests generate a synthetic package containing colored devices, a connector/arrow, nested shapes, a background page, and an embedded picture. Native tests verify actual pixel colors and page mapping, while focused tests cover source limits, cached fields, missing binaries, conversion failure, output limits, timeouts, partial pages, process handoff, persistent assets, and ACL checks.

The public sample used for manual review is `Network Diagram_start.vsdx` from the MIT-licensed sample repository [Aspose.Diagram-for-.NET at revision 542b4fa](https://github.com/aspose-diagram/Aspose.Diagram-for-.NET/blob/542b4fa43d9cc7d42d30ba9260839ca8999fbcc0/Examples/Data/Load-Save-Convert/Network%20Diagram_start.vsdx). Only the public sample was used; no Aspose software was installed or invoked. The sample and license are kept with the local evaluation artifacts, not bundled into Sauron.

Native arrow tests separately verify one-way and two-way solid arrowheads, including inherited line styles, in actual PNG pixels. The public topology sample has no effective arrowheads and therefore cannot validate directional arrows. The tested arrowhead geometry survives conversion, but libvisio renders the markers black even on colored lines; this is another fidelity limitation.
