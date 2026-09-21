# Visio ingestion

Sauron accepts modern `.vsdx` documents and renders whole pages with **libvisio + librsvg**, with Inkscape for embedded EMF images. Microsoft Visio, exported companion images, a second service, and a commercial SDK are not required. Legacy `.vsd`, `.vdx`, and macro-enabled `.vsdm` are explicitly rejected in this release.

## What is retained

The worker reads the VSDX package's native page names, shape text, groups, properties, and connector attachments. It also records saved begin/end coordinates (in the containing shape's coordinate system), line-end styles and sizes, and their shape/master/line-style provenance. Missing attachments and unsupported custom line ends stay unresolved; proximity does not create an attachment. It resolves available master text and cached string fields; it does not evaluate ShapeSheet formulas or refresh external data. Attachments establish drawing connections, not traffic direction, routing protocols, or failover intent. Original shape/page IDs remain in the source evidence.

For graphics, a temporary source copy materializes inherited text, saved string field values, text formatting, and resolvable inherited line-end settings that libvisio otherwise sometimes omits. Connector geometry, pictures, master definitions, and backgrounds remain source-defined. Existing line-end cells and formulas are preserved; only absent cells with known inherited saved values are materialized. The original source is unchanged. Libvisio outputs foreground pages followed by backgrounds; Sauron maps these back to their source page names, IDs, and package-order page numbers before attaching citations. Backgrounds are composited behind foreground diagrams and are also retained as separate pages.

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

Use an empty output directory. The utility runs the disposable worker, writes page PNGs, `source-text.txt`, and `report.json` (including source pages and connector evidence), and does not add documents to the database. Exit status 2 means no pages were rendered; inspect the report for the cause. The source is never uploaded externally by this utility.

## Validation

Tests generate a synthetic package containing colored devices, a connector/arrow, nested shapes, a background page, and an embedded picture. Native tests verify actual pixel colors and page mapping, while focused tests cover source limits, cached fields, missing binaries, conversion failure, output limits, timeouts, partial pages, process handoff, persistent assets, and ACL checks.

The public sample used for manual review is `Network Diagram_start.vsdx` from the MIT-licensed sample repository [Aspose.Diagram-for-.NET at revision 542b4fa](https://github.com/aspose-diagram/Aspose.Diagram-for-.NET/blob/542b4fa43d9cc7d42d30ba9260839ca8999fbcc0/Examples/Data/Load-Save-Convert/Network%20Diagram_start.vsdx). Only the public sample was used; no Aspose software was installed or invoked. The sample and license are kept with the local evaluation artifacts, not bundled into Sauron.

Native arrow tests separately verify one-way and two-way solid arrowheads, including inherited line styles, in actual PNG pixels. The public topology sample has no effective arrowheads and therefore cannot validate directional arrows. The tested arrowhead geometry survives conversion, but libvisio renders the markers black even on colored lines; this is another fidelity limitation.


## Integration evaluation (September 2026)

Five public VSDX files from the pinned Aspose sample repository rendered all nine
pages: the four-page network/rack sample, two-page BFlowcht flowchart,
AddConnectShapes, EditConnectorGeometry, and SetLineData. The flowchart visibly
preserves arrows on bent connectors. The line-style sample has severe text/layout
problems; producing a PNG is not a visual-fidelity pass. The geometry sample has
unresolved source line-end settings, now surfaced as warnings. No proprietary files
or Aspose runtime were used.

A sixth public source, [Wires and Wi-Fi's Cisco CVD icon document, revision 1.5](https://www.wiresandwi.fi/blog/visio-networking-icons-stencil-cisco-cvd-and-custom-icons),
extracts 1,024 shapes and now produces a page PNG after converting 141 embedded
EMF images with Inkscape. Icons appear, but overlapping/clipped text remains a
known layout defect. Multi-line labels in the native converter output lack SVG
line positioning; this occurs before EMF replacement. No content-fetch guard was
loosened. See [EMF ingestion](emf.md) for limits and analysis provenance.
The author's published preview is not Sauron's rendered output. This is an icon
collection, not a complete Cisco topology, and must not be represented as a passed
Cisco deployment-diagram test.

Synthetic native pixel tests cover end-only, begin-only, both-end and unmarked
lines, inherited styles and bent geometry. Source tests also cover master values,
explicit no-arrow overrides, style chains/cycles, custom unresolved markers, grouped
coordinates, dangling attachments, and preservation of original bytes. These tests
verify the covered cases, not universal Visio compatibility. WMF/general OLE rendering and
complex text layout remain gaps; supported EMF conversion is now available.


## Source text-box layout repair

`VISIO_REPAIR_TEXT_LAYOUT=true` restores source line breaks, word wrapping and
alignment using saved text-box dimensions, margins and inherited paragraph
settings. Cached instance coordinates, drawing scale and translation-only groups
are supported, as are styled text runs. Repeated labels require a unique source
text and coordinate match. Fontconfig and Pillow measure installed fonts, and the
SVG uses the measured font family to avoid a second fallback choice.

Adjacent overlapping text boxes on the same row are narrowed around their original
centers. Labels can use smaller preview fonts to fit; `VISIO_TEXT_MIN_SCALE=0.5`
sets the minimum relative size (allowed range 0.25–1.0). A 4-point floor also
applies, without enlarging source text that was already smaller. Set 1.0 to disable
shrinking. Both controls are available through the admin portal. This changes only
the preview: source text, saved source geometry, images and connectors are retained.

The repair preserves native converter output for ambiguous matches, rotated or
flipped groups, explicitly positioned spans and unsupported paragraph features.
Text still exceeding its saved box at the minimum scale is retained with a warning.
Font substitutions and these overflow cases still require visual review.

Compact, single-line, vertically centered badges with symmetric vertical margins
use visible glyph bounds for placement. This handles boxes whose padding consumes
the entire nominal text area, without changing top-aligned or asymmetrically padded
text. Original source settings are retained; only preview glyph placement changes.

The public Cisco sheet now has 244 repaired labels, including its rich-text title
and 64 centered compact labels such as SPINE and VTEP. 129 adjacent boxes are
narrowed and 169 labels use smaller preview fonts. Its 22 remaining minimum-size
overflows are flagged. The public network topology page has 56 repaired labels,
including previously clipped grouped callouts. Six public documents produced ten
PNG pages in the previous corpus run; the final badge pass was rerendered against
the Cisco sample. This confirms render coverage, not pixel-identical Visio fidelity.

### Diagram answers and Mermaid

Native Visio figure passages are checked against the saved extraction before
being treated as original evidence. OCR and visual-model descriptions remain
interpretations. This also works for existing ingestions; no re-embedding is
required. Diagram-focused answers omit unrelated knowledge-graph context when
native Visio evidence is available.

A basic Mermaid request under the partial-evidence answer policy can return a
source-derived component view of one named page, with up to 40 short native
labels and citations outside the code block. The selected page uses up to 32,000
characters of saved native evidence rather than only the retrieved search snippets.
Where available, the largest saved drawing group supplies the labels; duplicate
labels are collapsed. It is explicitly a partial view:
it does not infer network containment, connector attachments or traffic flow
from drawing groups or nearby labels. Requests for exact connections, multiple
pages or strict completeness use normal synthesis and the selected evidence
policy. Playground displays Mermaid source as a code block; clients with Mermaid
support can render it. The stored PNG remains the full visual reference.

### Paragraph layout in poster previews

Text repair follows the active Visio paragraph markers, including per-paragraph
alignment, spacing and standard hanging bullets. Unused paragraph styles no
longer disable wrapping. Margins and fixed line spacing follow the page scale.
The Zero Trust Overview poster now wraps its summary, poster links and nested
architecture labels within their saved text boxes. Source text and connector
geometry are preserved. Unsupported bullet types, transforms and first-line
indents still retain the converter output; minimum-size overflow is flagged.
