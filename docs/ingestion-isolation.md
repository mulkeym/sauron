# File extraction isolation in one container

Sauron still runs as one container and uses the existing `/app/data` mount.
File parsing, PDF table extraction, spreadsheet reading, OCR and figure
analysis run in a fresh Python subprocess. Only one extractor runs at a time;
indexing jobs can retain their existing concurrency.

The API streams uploaded files to disk and awaits extraction without blocking
its event loop. The child returns text, tables and figure provenance as bounded
JSON. The API retains ownership of embeddings, database writes and LightRAG,
so the change does not introduce competing writers or stale process-local
database registries. Both synchronous API uploads and queued uploads (including
admin uploads and downloaded connector files) use this boundary.

The supervisor starts the child with POSIX spawn rather than forking the
already-initialized API. This avoids copying LanceDB and other native runtime
state during child startup, which can otherwise exhaust a tight container or
LXC memory budget before the worker's own limits take effect.

A native crash, killed process, oversized result or timeout fails that file.
Queued ingestion records the error, cleans up partial writes and continues to
the next job. The synchronous API returns HTTP 422 for extraction failure.
An ordinary Python table-extraction error can still fall back to flat text,
but that fallback runs inside the child. Sauron never retries a crashed native
parser in the API process.

PDF extraction releases pdfplumber page caches after each page, including on
errors, and before starting OCR. Detected table boundaries are reused for
prose and figure placement, including borderless tables, so table detection
does not run again for each output. Scanned pages are copied into individual
temporary PDFs before calling Unstructured: its `page_numbers` keyword does
not restrict processing. This prevents the entire document from being OCR'd
again for each scanned page. The temporary reader is closed before OCR and
the one-page stream is closed on success or failure. Original page numbers,
rotation, and crop boxes are preserved. These changes use existing libraries
and add no model downloads. Memory exhaustion propagates to the worker's
failure handler instead of being swallowed by OCR or layout fallbacks.

## Limits

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `EXTRACTION_TIMEOUT_SECONDS` | `1200` | Maximum runtime of one extractor, including OCR/vision |
| `EXTRACTION_MEMORY_MB` | `4096` | Maximum memory growth allowed during extraction |
| `EXTRACTION_MAX_RESULT_MB` | `32` | Maximum JSON result accepted by the API |
| `EXTRACTION_WORK_DIR` | `data/extraction` | Private temporary input/result directories |

On Linux, the supervisor watches total memory charged to the container while
the worker runs. It honors the configured growth limit and also reserves one
quarter of available memory (at least 512 MiB) for the API. Sauron checks both
the cgroup limit and `/proc/meminfo`; the latter covers Docker running inside a
memory-limited LXC where the inner cgroup can appear unbounded. The supervisor
kills the worker process group before it consumes that reserve. This uses
resident container memory rather than an address-space limit because parser
libraries can map several gigabytes of virtual files while using much less RAM.
If an OCR model cannot fit, the upload fails; adjust the container and extraction
budgets together. Systems without cgroup usage metrics still get crash isolation
and timeouts, but cannot enforce the aggregate memory ceiling.

This is process isolation within a shared container. It does not guarantee
survival of host-wide or container-wide OOM, or protect the API from failures
in indexing/embedding code that still runs there. Full resource separation
would require a separate container. The job-status queue is still in memory;
this change does not make queued jobs survive an API restart.

Timeouts, cancellation and shutdown kill the worker's process group, including
OCR/rendering descendants, and reap the child. Temporary inputs, settings and
results are removed afterward. Compose enables `init: true` to reap orphaned
grandchildren. For a manual `docker run`, include `--init`.

## Deploy

Rebuild and recreate the existing API service:

```sh
docker compose up -d --build api
```

No additional service, port or volume mapping is required. Keep the existing
`app_data:/app/data` mapping (or your host-folder equivalent). Limits can be
set in `.env` using the variable names above.

## Validation

The regression tests exercise real SIGSEGV, SIGABRT and SIGKILL child exits,
timeout and cancellation, a healthy follow-up import, and API health while a
queued worker dies. Other tests verify that PDF tables and positioned figure
metadata survive the process/JSON boundary and that oversized results are
rejected.

```sh
python -m pytest tests/test_ingestion/test_isolation.py
python -m pytest tests/test_ingestion/test_pdf_resources.py tests/test_ingestion/test_pdf_extract_io.py
```

PDF resource tests use real PDF splitting and table extraction, with the OCR
model call replaced by a test double. They verify single-page inputs, original
page numbers, cleanup on success and failure, and reuse of table boundaries.

Runtime references: [Python resource limits](https://docs.python.org/3/library/resource.html)
and [Docker Compose init](https://docs.docker.com/reference/compose-file/services/#init).
