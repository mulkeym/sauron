# Diagnosing an empty knowledge graph

The progress label “extracting relationships” identifies an active stage. It
does not confirm that the model returned parseable records or that storage
succeeded. Previously Sauron could report completion after a LightRAG failure:
the adapter returned an error string, and count-read errors became `(0, 0)`.
LightRAG can also store a per-document failure without raising from `ainsert`.

The adapter now checks document status, rejects empty model text, propagates
insertion/persistence/count errors, and the queue retains graph failures in
job warnings. Searchable document passages remain available. Completion counts
are *new* global entities/relationships, so reusing existing entities can
legitimately add zero. If the whole graph remains empty, the job warns explicitly.
These global deltas are approximate with concurrent ingestion.

## Checks inside an air-gapped installation

1. Record the Sauron image revision, serving backend, and installed LightRAG
   version. Earlier Docker builds installed an open-ended LightRAG constraint,
   so a lockfile did not establish the installed version. Updated builds pin
   `lightrag-hku==1.5.7` in both requirements files.

   ```sh
   docker exec sauron python -c 'from importlib.metadata import version; print(version("lightrag-hku"))'
   ```

2. Inspect the container logs around the import for `LightRAG insert failed`,
   `KG extraction failed`, `LLM output format error`, timeouts, authentication,
   context-length, embedding, and storage errors. Also inspect the document's
   `status` and `error_msg` in `data/lightrag/**/kv_store_doc_status.json` inside
   the container. Keep raw logs and status files within the protected network:
   they may include document information.

3. Ingest a new small UTF-8 `.txt` file with explicit synthetic relationships:

   ```text
   KGProbe_Router_Alpha connects to KGProbe_Router_Beta over an MPLS link.
   KGProbe_Router_Beta is located at KGProbe_Site_West.
   KGProbe_Operations_Team maintains both routers.
   ```

   If that also produces an empty graph, investigate the model adapter/output
   format, embeddings, library compatibility, and graph storage. If it succeeds,
   compare the affected document's extracted text and chunk budgets. This probe
   creates normal document/graph records; delete the test document afterward if
   desired. It does not require sharing protected source files.

Graph extraction uses LightRAG's model adapter with Sauron's configured text
model endpoint. The final-answer reasoning switch does not control this path.
A Gemma model name alone cannot determine whether reasoning output, formatting,
a timeout, or another error caused a particular import to fail. Do not clear
the graph or extraction cache before inspecting the failure evidence.

## LightRAG 1.5.7 compatibility

LightRAG 1.5.7 removed the singular `get_docs_by_status` method from its status
store. Sauron now uses `get_docs_by_statuses(list(DocStatus))` for both document
listing and pre-insert orphan cleanup. All current statuses are included, and
listing errors propagate instead of looking like an empty corpus. Pre-insert
cleanup must succeed before submitting a document; a failed cleanup no longer
falls through to a misleading duplicate/no-work result.

The old missing-method warning identifies a cleanup compatibility defect; it
alone does not establish why entity extraction returned zero. Runtime tests use
real LightRAG storage and synthetic model/embedding output to verify successful
extraction (two entities, one relationship), internally recorded model failures,
and a non-record model response that produces an empty graph. These tests do
not exercise the protected installation or its Gemma server.

## Ignore TLS errors with OpenAI SDK 3.16.2

OpenAI SDK 3.16.2 uses `httpx2` for its default HTTP transport. Sauron's older
shared TLS hook patched only `httpx`, so LightRAG could continue verifying an
untrusted endpoint's certificate even with **Ignore SSL certificate errors**
checked. Ordinary answer requests use a separate HTTP path and could still work.

The shared hook now covers both HTTP libraries. Newly constructed SDK clients
read the current `ssl_verify` setting, including when verification is re-enabled.
Explicit per-client verification options retain precedence; already-created
HTTP clients retain their TLS context. LightRAG creates fresh clients per attempt.
Future builds pin OpenAI SDK 3.16.2 as well as LightRAG 1.5.7.

A local HTTPS fixture with an untrusted certificate reproduced
`RetryError ... APIConnectionError` with the checkbox enabled before the fix.
After the fix, real LightRAG/SDK calls succeed with the checkbox enabled, reject
the untrusted certificate when verification is enabled (including after toggling
back), and accept an explicitly trusted private CA with verification enabled.
This verifies the transport defect without sending any protected document data.
The updated application still needs deployment and a new import to verify the
protected installation's complete ingestion path.


## Recover previously uploaded documents

After installing the updated build and correcting the model connection:

1. Wait for active ingestion to finish.
2. Open **Knowledge Graph → Rebuild from uploaded documents**. This queues all
   current catalog documents, across all datasets, using the existing ingestion
   queue. It does not require the original files to be uploaded again.
3. Expand **Rebuild progress and errors**. Keep Sauron running until completion,
   then reload the page to view the updated graph. Correct any reported model or
   storage errors and retry. Queue history is in memory; after an application
   restart, explicitly start another rebuild rather than assuming it completed.

Recovery uses all retained medium-tier indexed passages and figure descriptions,
with pagination (including documents over 200 passages). It strips the repeated
synthetic document summary and deduplicates identical passages. It does not
reparse originals or recreate text omitted during initial ingestion. Documents
without retained passages report an error; structured spreadsheet files are
skipped. Existing documents, ACLs, source files, diagrams, and search embeddings
are not deleted or reindexed. Graph embeddings and model extraction are rerun.

Each document's old LightRAG status/graph contribution is removed under the
insertion lock before rebuilding; associated cached model extraction is cleared
so a previously empty extraction can be corrected. Orphan/duplicate records
from earlier failed deletes are reconciled first. Other documents' graph
contributions remain. During rebuilding, the graph can be incomplete; if a
rebuild fails, that document may temporarily lack graph data, but its ordinary
search results remain available.

Deletion also clears associated extraction caches so a later upload cannot
reuse an old empty extraction. It validates LightRAG's returned result and verifies that its status
record is gone. Admin deletion waits for graph cleanup and reports HTTP 503 if
cleanup fails instead of silently returning success. Metadata/search deletion
may already have completed; retry deletion or run graph recovery after fixing
the reported failure. A retry of a failed ingestion ID resets its failed record
before resubmitting, because LightRAG otherwise treats the retry as a duplicate.

Regression cases use actual LightRAG 1.5.7 local storage to test successful
insert/delete/reinsert, retry after model failure, explicit rebuild after empty
extraction, and recovery from a stale failed primary plus duplicate upload record.

Graph-only rebuild jobs run one document at a time, so a document waiting for
its turn does not consume its extraction timeout. Rebuild completion reports
that document's entity/relationship totals (including shared entities), rather
than a global before/after delta that can be zero when replacing existing data.
The status panel shows the latest rebuild attempt for each document.

### Slow or failing individual chunks

Graph model calls default to a **4,096-token output cap** and a **180-second total
model-call deadline**, including the OpenAI adapter's retries. The shorter of the
graph deadline and general model timeout applies. These controls are available in
**Settings → All Settings → Document processing** (`kg_llm_max_output_tokens`,
`kg_llm_timeout_seconds`, and `kg_llm_disable_thinking`). They are independent of
final-answer thinking. Thinking is disabled using the configured reasoning
adapter: select **vLLM template** for a compatible local Gemma server. Unsupported
providers log a warning rather than receiving unsupported request parameters.
Model servers that require reasoning cannot guarantee thinking is off.

Sauron's adapter schedules chunks independently through LightRAG 1.5.7. A failed
model call does not cancel the remaining chunks. Successful chunks are merged;
the document is marked **failed/incomplete**, with failed chunk IDs and errors in
its LightRAG status metadata (`sauron_failed_chunks`). The queue displays the
incomplete warning and retained entity/relationship counts. Truncated model
outputs are failures, not complete cached extractions.

Retry using **Rebuild from uploaded documents**. For an incomplete document, the
retry resets its graph/status anchors but retains successful extraction caches;
only uncached extraction calls need the model again. Merge work may call the
model again. A rebuild of an already complete document still clears its old
extraction cache. Original uploads and the search index are unaffected.

The overall document deadline still applies. User cancellation and storage
failures stop processing and drain active workers; the continuation behavior is
not a promise to finish every chunk after a whole-job timeout. This integration
uses a private LightRAG scheduling hook, so rerun the real-storage regression
tests before upgrading the pinned dependency.
