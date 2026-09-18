# Answer profiles

Open **Settings → Answer Profiles**. General guidance, Deployment guidance, and
Technical troubleshooting presets are included. General guidance is active
initially. Profiles control retrieval and answers; source-document approval,
ownership, and versions remain outside this feature.

## Workflow

1. Select a profile or **Create a copy** to start another working draft.
2. Edit answer instructions, routing guidance, retrieval controls, and response
   policies. **View the full prompts** shows the composed prompts; refresh after
   edits. Automatic routing also receives the user's accessible table catalog.
3. Enter a sample question and test ACL groups. Empty groups grant no document
   access. Dataset ID 0 means all datasets accessible to those groups.
4. **Run draft preview** tests current form values, including unsaved edits,
   against the production graph. It displays the response, selected strategy,
   evidence supplied to the model, and passages cited. It uses the configured
   model and data connections, but does not publish, reuse/write cached answers,
   or record strategy-training feedback.
5. **Save draft** persists changes without affecting live answers.
6. **Publish saved draft** creates an immutable revision and activates it for new
   requests. A preview is available but is not a mandatory approval gate.

One profile is active globally. It applies to generated MCP answers, REST query
endpoints, asynchronous jobs when they start running, OpenAI-compatible chat,
and the normal admin playground. Raw retrieval tools remain raw retrieval tools;
a client-supplied tool argument does not select an admin profile.

Each request captures its profile before the cache check and retrieval. Publishing
mid-request cannot change that request's profile. Draft edits do not invalidate
live answer caches; changing the active profile/revision does.

## Controls

Answer instructions supplement shared team instructions and fixed grounding and
citation rules. Shared instructions remain under **Answers & Evidence**.

Routing can use automatic classification or a fixed existing strategy. Fixed
routing bypasses the classifier and learned strategy overrides. Adding executable
strategies requires code changes; automatic routing guidance is editable here.

| Retrieval depth | Lookup candidates | Discovery results per search | Subquery results | Neighbor window |
|---|---:|---:|---:|---:|
| Focused | 15 | 50 | 5 | 1 |
| Balanced | 30 | 250 | 10 | 2 |
| Thorough | 60 | 500 | 20 | 3 |

Query expansion and date matching can add documents. These limits control search
size, not completeness. Broader searches can take longer.

Other controls include graph enrichment, structured table lookup, learned
strategy overrides, and a maximum of 0–8 extra subqueries. Disabling structured
lookup also disables SQL escalation in lookup/sweep/cross-reference strategies;
fixed analytical routing requires it. Learned overrides are disabled in the
presets and also require the global Strategy Memory setting when enabled.

Clarification can ask about selected missing details—platform, software version,
environment, and site—or request a supported answer with caveats. Insufficient
evidence can produce a cited partial answer identifying gaps, or abstention when
the complete requested answer is unsupported.

The model returns an answer, clarification, or insufficient-evidence status.
Clarification questions use fixed application text for the selected details and
do not require fabricated citations. Abstentions also use application text.
Factual answers still require valid references to supplied evidence. Models that
return plain text can produce a cited answer, but uncited plain-text clarification
is rejected. Citation validation does not establish semantic entailment; routing
and evidence-sufficiency judgments still depend on the configured model. Test
representative questions before publishing.

The normal playground uses the shared synthesis function and displays the
completed validated response, so the model's JSON envelope is not displayed as
streaming answer text.

## Rollback and persistence

Choose a published revision, inspect its settings, and **Activate selected
revision** to roll back. The working draft is preserved. Published revisions and
activation history remain available. Rollback restores the profile settings;
it does not undo shared team instructions, model settings, or source/index changes.

Drafts, revisions, the active selection, and activation history persist in
`data/answer_profiles.sqlite3`, covered by the existing `/app/data` volume.
No additional container, service, dependency, or model is required.

SQLite transactions and version checks prevent stale admin tabs from overwriting
newer changes. Reload before retrying a conflict. The profile APIs inherit the
existing admin-session and same-origin protections.

## Verification

441 targeted regression tests passed across agent strategies, retrieval,
generation, MCP, admin, query APIs, authentication, and settings. New tests cover
draft isolation, publication/rollback/reopen, stale edits, validation, pinned
requests, cache scope, routing controls, citation handling, and the real preview
graph with synthetic evidence. The run includes existing test-key, deprecation,
and async-mock warnings.

Browser checks covered unsaved preview, save, publish, rollback, clarification,
empty access, error feedback, keyboard focus, and desktop/narrow layouts in an
isolated instance. No protected documents or live production model endpoint were
used. No running deployment was rebuilt or changed.
