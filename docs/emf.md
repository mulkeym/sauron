# EMF diagram ingestion

Sauron accepts standalone `.emf` files and EMF images embedded in modern `.vsdx`
documents. Inkscape converts their drawing records to PNG inside the disposable
extraction worker in the existing container. No Visio installation, separate
service, extra model weights, or manual export is required for VSDX uploads.
WMF, arbitrary OLE objects, and universal EMF/EMF+ fidelity are not promised.

Standalone diagrams produce a persistent full PNG and preview. Visio-embedded
EMFs are replaced in the converter's SVG with transparent PNG data images;
placement, transforms, dimensions and clipping attributes are preserved. Repeated
identical images reuse a disk-cached conversion. Embedded PNGs are composed into
the page figure rather than becoming unrelated search results for each small icon.
The original source bytes are never changed.

OCR extracts labels; the existing configured **vision-capable** LLM describes the
rendered standalone drawing or Visio page containing EMFs. Source information,
OCR text, and visual interpretation are retained separately in figure metadata.
Every indexed analysis chunk identifies its evidence type. A visual description
is not a native connector fact. Native VSDX connector attachments and saved
arrowhead evidence remain available alongside the visual interpretation.

Model or OCR failure retains the image and available evidence with warnings.
A standalone conversion or image-storage failure fails that ingestion clearly.
An embedded conversion failure withholds the affected page PNG and retains its
native source text, rather than returning a diagram with a silently missing icon.
Existing figure ACLs, citations, retrieval and MCP image attachments apply.
Original `.emf` bytes are retained when private original storage is configured and
can be downloaded as an attachment through the existing OpenWebUI proxy using
its already-supported `application/octet-stream` type; no OpenWebUI format change
is required. With original storage disabled, only PNG assets and extracted data
are retained, as for other supported formats.

## Controls

All controls are available in **Admin > All settings > Document processing** and
as environment variables:

| Setting | Default | Meaning |
|---|---:|---|
| `EMF_ENABLED` | true | Enable standalone and embedded conversion |
| `EMF_MAX_INPUT_MB` | 32 | Maximum individual EMF source size in MiB |
| `EMF_MAX_CONVERSIONS_PER_DOC` | 256 | Maximum distinct native conversions |
| `EMF_TIMEOUT_SECONDS` | 120 | Total native EMF conversion time per document |
| `EMF_RENDER_MAX_EDGE` | 4096 | Standalone PNG longest edge in pixels |
| `EMF_EMBEDDED_MAX_EDGE` | 1024 | Embedded icon longest edge before page composition |
| `EMF_OCR_TIMEOUT_SECONDS` | 30 | OCR timeout per analyzed diagram |
| `EMF_VISION_ENABLED` | true | Use the existing configured multimodal endpoint |

Existing figure analysis count, image storage count/bytes, pixel limits, vision
response/timeout limits and extraction memory/time limits also apply. Disable
EMF vision analysis to use only PNG and OCR. Standalone EMFs require figure
extraction and storage to be enabled. Re-ingest or use figure backfill to add
images to existing documents; this does not rewrite previously ingested figures.

Inkscape is installed without recommended packages in a separate Docker layer.
A Debian slim test image measured approximately 323 MiB of additional installed
packages for Inkscape plus Tesseract. Tesseract and some dependencies are already
in Sauron's runtime, so this is not an exact production image delta. CI's existing
under-1-GB layer check remains unchanged. No full production image build or
vulnerability scan is implied by this dependency measurement.

## Validation and remaining limitations

The public Wires and Wi-Fi Cisco CVD icon sheet now renders through the isolated
Linux worker: **141 embedded EMF images converted, one page PNG retained**.
Individual icons are visible. This is an icon collection, not a complete network
topology. Direct libvisio output has the same image placements and text elements
as the prepared source; multi-line text is emitted without positioning new SVG
lines. The text-layout repair now restores 244 labels, including the rich-text
heading, and separates adjacent overlapping label boxes around their saved centers.
It supports styled runs and translation-only groups, with bounded font fitting.
22 labels still exceed their saved boxes at the minimum preview size and are
flagged for review. See [Visio layout settings and limitations](visio.md#source-text-box-layout-repair).
This does not certify all glyphs, icons, fonts, cropping, layers or connectors as
Visio-identical.

Synthetic tests cover a rectangle/right-arrow EMF, a Visio topology with an
embedded EMF and native connectors, placement/transform preservation, malformed
records, conversion budgets, original retention, OCR/vision failure, and separately
attributed searchable evidence. A standalone topology exported to EMF was also
rendered through the isolated worker with real Tesseract OCR. OCR recovered branch
and router labels but missed the WAN label, illustrating that OCR is incomplete.
Vision integration is tested with controlled responses and failures; real model
quality was not evaluated in this run.

Inspect a local file without indexing it:

```sh
python scripts/inspect_visio.py diagram.emf --output /tmp/emf-inspection
```

The output includes full/preview PNGs and a JSON report. This inspection uses the
same configured extraction, OCR, vision and asset limits as ingestion.

## Full-frame EMF+ wrapper clipping compatibility

The existing Inkscape renderer is retained. A narrowly gated preparation step
recognizes dual EMF+ files whose sole drawing is a metafile image mapped exactly
to the outer frame, with an explicit full-frame EMF+ clip. Their GDI fallback can
repeat a device-sized frame rectangle under a world transform, incorrectly
clipping the converted icon. The temporary input retains the initial frame clip
and all drawing/transform records, omitting only later copies of that same frame
intersection. The original file and Visio SVG placement attributes stay unchanged.

Unknown EMF+ records, extra drawing operations, partial/rotated/reflected placement,
nondefault image effects, malformed records and other GDI clipping/mapping
operations do not qualify. Intentional smaller clips and ordinary EMFs stay on
the native path. This is a compatibility correction for a verified wrapper pattern,
not a general EMF+ renderer or a promise that all clipping defects are resolved.
The applied correction is recorded in figure rendering warnings.

The public Cisco sheet exercises 49 corrected wrappers. Visual inspection shows
restored switch borders, router circles, arrowheads, key/lock outlines and cloud
icons. Synthetic native tests compare a repaired wrapped arrow pixel-for-pixel
against an unclipped reference and verify that an intentional crop stays cropped.
No renderer dependencies or container packages were added by this correction.

Format references: [EMF+ image placement](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-emfplus/9bad3867-71b0-46fb-9b90-f7d8fb37ff76)
and [EMF rectangle clipping](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-emf/13cd0c98-d4e9-4ca7-a79d-58055bf45c79).
