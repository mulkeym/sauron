# Synthesis: lead with a direct answer

**Date:** 2026-05-31
**Status:** Approved design, pending implementation
**Area:** `src/agent/synthesizer.py` (prompt constants only)

## Problem

Answers feel disorganized — they bury the actual answer. The synthesizer's
prompts are tuned for exhaustive enumeration ("include EVERY unique item",
"list all 50", "prioritize listing items over adding descriptions") with no
notion of an opening answer, so the model dives straight into a data dump.

Concrete example — "what are the pay rates for a gs-13?" produced a wall of
rates broken out by every executive order and every year, with no direct
bottom-line answer at the top.

The user's single priority (confirmed): **lead with a direct answer.** The
supporting detail and completeness should stay; only the ordering/lead is wrong.

## Approach (chosen: A — prompt-only)

Rewrite the two prompt constants in `src/agent/synthesizer.py`
(`SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`) to impose a two-part answer structure
and re-scope the completeness rules to the *detail* section so they reinforce
the lead instead of fighting it.

Rejected alternatives:
- **B — query-type-aware prompts:** branch synthesis by classification. Adds
  branching/maintenance for a behavior wanted uniformly. YAGNI.
- **C — two-pass synthesis:** generate, then a second LLM pass to prepend a
  summary. Extra latency/cost for what one pass can do.

No structural/code change: same `synthesize_answer` function, same retrieval,
same citations. Only the prompt text changes.

## Design

### Answer shape (uniform, adaptive — not a rigid template)

1. **Lead:** a direct 1-2 sentence answer to the exact question — the specific
   value, range, count, or finding — before any breakdown. For a list-type
   question the opening states what the list contains and how many
   (e.g. "Twelve DHA contracts were awarded in January 2026:").
2. **Detail:** the complete supporting information. Stays THOROUGH and COMPLETE
   — include all relevant info, list EVERY instance, dedup by identifier, cite
   sources, use bullets/numbered lists, group under short headings when helpful.
3. Don't restate the same facts in both the lead and the detail.

### `SYSTEM_PROMPT` (revised)

Replaces the flat "Rules" list with: the grounding/citation rules, then a
"Structure every answer" block (lead → detail → no-restate), then the existing
"output only the final answer, no reasoning" guard.

### `USER_PROMPT_TEMPLATE` (revised)

Keeps the existing completeness + deduplication guarantees verbatim in intent,
but reframes them under "FIRST a direct answer, THEN the complete detail (list
all 50, dedup by identifier, cite sources)."

## Testing / verification

- **Existing synthesizer unit tests stay green** (they cover citations,
  context-building, reasoning-strip — not prose quality).
- **New unit test:** assert both prompt constants carry the "direct answer
  first" framing, guarding the intent against silent regression. (Prose quality
  itself can't be unit-tested without an LLM; this guards the requirement.)
- **Live verification (deployed):**
  - GS-13 pay question opens with a direct rate answer, then the detail.
  - A contracts list question opens with a count/summary line, then the list.
  - Full suite shows no new failures vs the 15-failure baseline.

## Out of scope

- Trimming/over-listing, redundant-source consolidation, formatting overhaul
  (user did not flag these; only "no direct answer first").
- Retrieval changes.
