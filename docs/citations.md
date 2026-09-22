# Readable answer references

Answers show the filename first, followed by available page, slide, section or diagram details. If these are missing, a document passage uses a one-based passage number (`chunk_index + 1`). Derived summaries and database query results are labelled as such; page/section details are never guessed.

Example: `[sdwan-for-gov.pdf — passage 127]`.

In the admin playground, the reference links to its citation card and opens the supporting passage, including keyboard activation. Existing HTTP(S) source URLs remain available on the card. Other clients receive a Markdown source link when a source URL exists, or a readable bracketed label otherwise. Direct PDF URLs gain a page fragment only when page metadata exists and the source URL has no fragment. No public file/chunk URLs or download credentials are invented.

The presentation layer covers fresh and cached playground answers, previews, REST sync/async results, OpenAI-compatible responses, and MCP ask/summary/comparison tools. REST and MCP citation records include `display_label` alongside unchanged canonical `evidence_id`, document IDs and source provenance. Internal validation, cache records and source text keep their original identifiers. Formatting does not resolve unknown IDs, change quotations, or convert literal code markers into references.

Existing rendered browser results need a new query to obtain the updated presentation; cached answers are reformatted when served.

## MCP → OpenWebUI native citations

`tool_answer_from_documents`, `tool_summarize_topic`, and `tool_compare` now return
`sauron_citations_version: 1`, the human-readable answer, a canonical
`answer_with_evidence_ids`, and the complete `citations` array. The canonical
field is for client adapters; it should not be displayed alongside the readable
answer. Evidence IDs identify passages, not filenames or browser routes.

The companion OpenWebUI adapter assigns native citation numbers across tool
calls and other sources. Each source card contains the exact returned snippet
and `display_label`; an original-file action uses OpenWebUI's authenticated
proxy. No playground fragment, API key, or model-provided download URL is used.
The adapter supports both native and legacy tool calling. Existing clients can
continue displaying the readable answer without implementing the adapter.

Other MCP clients can call `tool_get_cited_passage` with the exact `doc_id`,
`source_revision`, `evidence_id`, `chunk_index`, `chunk_size_tier`, and
`start_char` from a citation. It checks current catalog permissions, dataset
status, revision, indexed location, and evidence hash. It never substitutes a
newer or similarly named passage. Generated/query evidence may not correspond
to a retrievable indexed passage; its returned snippet remains the answer's
supporting evidence snapshot. Missing revision metadata cannot authorize an
original-file link or an exact revision read.

## Listing and showing stored diagrams

Requests such as “list the diagrams for sdwan” and “show me a topology diagram”
use direct figure discovery in the shared answer pipeline. They bypass model
classification, catalog SQL, graph enrichment and answer generation. The result
lists up to ten verified stored figure candidates with filename/page citations;
preview attachments still respect the configured image limit and answer profile.
These are ranked candidates, not an exhaustive inventory or a claim that every
image is a topology. Missing image assets are excluded. Access groups and selected
document/dataset scope still apply. Requests to explain or compare diagram content
continue through the normal grounded answer pipeline.
