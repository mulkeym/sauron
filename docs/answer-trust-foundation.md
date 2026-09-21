# Answer trust foundation and admin configuration

This is the first implementation stage of the SD-WAN answer-system review. It preserves the single-container deployment and adds no Python or model dependencies. The earlier ingestion-isolation and PDF changes remain in place.

## Admin portal

- **Settings → Answers & Evidence**: edit team/domain instructions, inspect the effective system prompt, and configure cache mode, lifetime, and the applicability threshold. The fixed evidence rules remain visible and apply to all domain instructions.
- **Settings → All Settings**: every field in the application's Settings model is available through a labeled, typed control. This includes extraction worker limits, model settings, integration credentials, and storage configuration. Existing focused settings pages remain available.
- Secrets are blank/masked in All Settings. Blank means keep the existing secret; optional credentials can be explicitly cleared.
- Invalid submissions do not mutate live configuration or write the settings file. Successful saves use atomic replacement and owner-only file permissions.
- Saved settings live in `data/settings.json` on the persistent volume and override environment defaults on restart.
- Storage paths, cached models, transport configuration, and worker sizing are saved as pending changes until restart. All Settings displays pending values and the save response lists the restart requirements. Changing embedding models also requires rebuilding the document index.
- Admin API endpoints require an authenticated admin session; cross-origin admin API requests are rejected.

## Query and evidence behavior

The graph resolves accessible document IDs from the authoritative metadata catalog before retrieval. An empty group list, empty dataset, or empty allowed-document list no longer means unrestricted access. Date matches intersect the existing document scope. Catalog queries and document-backed structured tables use the same scope. Raw MCP document/meeting searches also consult current metadata permissions.

Graph enrichment now returns filtered graph data instead of generating an intermediate answer. A graph record must identify only allowed source files. Mixed permitted/restricted records, missing provenance, and filenames that are ambiguous across allowed and denied documents are excluded. This deliberately reduces recall until immutable document IDs are available throughout the graph.

Evidence packing assigns stable `[E…]` identifiers to the exact passages admitted to the context. Original passages precede derived summaries. An oversized passage does not discard later passages that can fit; omitted passages and SQL rows produce warnings. No model call occurs when nothing usable fits.

Generated answers must reference supplied evidence identifiers. Invalid references and responses without usable references produce a bounded failure message. Returned citations contain only referenced evidence; snippets now hold the entire admitted passage. Source URLs, existing page/section locations, source kind, tier, and character positions survive REST, async, MCP, and cache serialization. The playground provides expandable supporting passages.

Reference validation establishes that a cited passage was supplied to the model. It does **not** establish that the passage semantically entails every claim. Team evaluation remains necessary. Missing page/section metadata in previously indexed prose is not reconstructed by this change.

## Cache policy

The default is **off**. Optional exact and semantic modes require matching document/catalog revisions, current group and dataset scope, retrieval mode, LanceDB index revision, model/retrieval settings, and prompt configuration. Any catalog change invalidates reuse conservatively, even when an unrelated document changed. Entries also expire according to the configured maximum age.

Semantic reuse additionally requires an affirmative applicability judgment above the configured threshold. Missing or failed checks fall back to retrieval. Older entries without a scope revision are ignored. Responses with evidence warnings, missing citations, or derived/SQL citations are not stored for reuse.

## Client compatibility

`/v1/chat/completions` now requires an application API key **and** a user identity, using the same authentication resolver as MCP:

- `X-API-Key` plus `Authorization: Bearer <Sauron user JWT>`, or
- `X-API-Key` plus configured OpenWebUI signed identity forwarding and group headers, or
- `X-API-Key` plus username/group headers when an administrator explicitly enables **Trust OpenWebUI user headers**. This accepts the connector's assertions without a shared JWT secret. See [MCP setup](MCP_OPENAPI_SETUP.md).

An API key alone no longer implies the `ALL` group. Existing API-key-only clients must supply user identity. Conversation-history handling and streaming compatibility are scheduled for the response-mode stage; they are not changed here.

MCP `tool_lookup_document` accepts `offset` and `limit` (1–200). It reads by exact document ID or a unique authorized filename without a global similarity search. Follow `next_offset` until null. `complete` is true only when one response contains all indexed passages. The representation is indexed medium/table-row passages, not the original file bytes; repeated overlap may exist. Use returned chunk positions when reconstructing reading order.

## Validation

392 targeted regression tests passed across agent strategies, retrieval, generation, MCP, admin, authentication, query APIs, and settings. New tests cover source revocation/deletion/editing, empty scope, mixed-source graph records, cache expiry and applicability failures, exact context/citation alignment, direct reads beyond the former search limit, and protected admin settings saves. An additional 63 ingestion/API regression tests passed, covering the earlier extraction and PDF work. The runs include warnings from short test JWT keys, deprecated APIs, and async test-fixture cleanup; these were not promoted to errors.

The two new admin pages were inspected in an isolated localhost browser preview using the existing stylesheet. Saving team instructions and retaining them after reload were verified. No protected documents or live model quality evaluation were used. No production container was rebuilt or deployed.

## Next stages

Source-document lifecycle management is deferred to the future remote document
system; local approval/ownership/version workflows are not planned.

Editable answer profiles, retrieval controls, preview, publication and rollback
are now implemented; see [Answer profiles](answer-profiles.md).

1. Shared answer/evidence/both response modes, conversation resolution, and one
   final answer writer for the OpenWebUI integration.
2. Consistent production telemetry and a team-reviewed evaluation set before
   enabling adaptive strategy changes.

## Graph-guided source retrieval

When the selected answer profile enables graph enrichment, lookup, procedure and
troubleshooting queries can use authorized graph entity names to search the
original documents again. For example, a graph match for **Admin-Tech File** can
expand a question about a **tech file** without requiring a hardcoded synonym.
This runs after the initial document and graph searches have both finished.

The follow-up is limited to one search, up to three terms and three source
documents. It uses the existing technical follow-up result and character limits.
Document permissions, dataset selection, filename ambiguity and source revisions
are checked again. Only original medium text passages containing a selected term
are promoted into the evidence pack; source checks use layout-normalized terms,
not a claim that the graph has established semantic equivalence. Recovered
passages remain available through final reranking. The request state records
selected terms, document IDs and passage counts in `graph_retrieval_trace`.
Disabling graph enrichment disables this follow-up too. No re-ingestion is needed.

Quotation validation also recognizes source-verified inline code as literal
support, alongside quoted prose. A filename appearing only in citation metadata,
a generated description, or a fenced generated code block does not qualify.
This avoids rejecting a documented filename pattern simply because the model
formatted it with backticks instead of quotation marks. Literal support and
citation validation do not certify semantic entailment of every generated step.
