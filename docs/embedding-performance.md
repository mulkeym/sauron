# Local CPU embedding performance

Open **Admin → Settings → Models → Local embedding performance**, or **All Settings**.

- **Batch size** (`embedding_batch_size`, default 4): maximum passages per local inference batch, for every chunk tier. The old hidden limit of four and fixed 4/4/2/1 tier limits are removed. Longer passages require more memory. Explicit smaller caller batches still take precedence.
- **CPU threads** (`embedding_cpu_threads`, default 0): 0 detects CPU availability from affinity and container quota. Explicit values are capped to available CPUs.
- **Inter-op threads** (`embedding_cpu_interop_threads`, default 1): parallel model operations. Increasing both thread settings can reduce throughput through oversubscription.
- **Memory budget** (`embedding_worker_memory_mb`, default 4096 MiB): bounds the Linux worker's resident memory and container growth. Container protection also reserves at least 512 MiB or one quarter of total memory for other work. Monitoring stops the worker on a breach; it is not a kernel memory reservation.
- **Request timeout** (`embedding_worker_timeout_seconds`, default 1200): includes model loading for a cold worker, excludes queue waiting.
- **Idle timeout** (`embedding_worker_idle_seconds`, default 300): releases the model after inactivity.

Batch and timeout updates apply to the next request. Changes to CPU threads, memory budget or idle timeout recycle the worker before the next request. An active request keeps its settings. Model/provider changes still require the normal application restart. Settings persist in the existing settings datastore.

One worker per API process handles local embedding requests serially, including document ingestion and query embeddings. It stays in the same container but outside the API process. Multiple file imports therefore share a model instead of loading competing copies. On a confirmed memory violation, the worker is discarded and the request retries with half the batch size, down to one. All attempts share the original request timeout. The smaller cap is retained for subsequent requests until the embedding configuration changes or the API restarts; the saved admin batch setting remains the requested maximum. A crash, invalid result, timeout, or memory failure at batch size one fails the request; the next request starts a fresh one. There is no fallback into the API process. Application shutdown stops it; on Linux, parent death terminates it even during native inference.

The Models page shows worker state, available thread counts and the last successful request's batch, thread counts, elapsed time, passages/second, queue wait and warm/cold status. Throughput includes IPC and cold model loading when applicable. Compare similar passage lengths and warm requests while tuning; these figures do not measure OCR, metadata generation or knowledge-graph processing. Changes do not alter the embedding model, prefixes, vector dimensions or truncation rules and do not require re-ingestion.

Existing saved batch sizes are honored. For example, a stored value of 64 previously clamped to four will now permit 64. Memory recovery can reduce the effective batch automatically. If even batch size one fails, check container headroom and the model worker memory budget.
