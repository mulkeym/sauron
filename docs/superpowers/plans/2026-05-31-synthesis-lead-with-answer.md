# Lead-with-Answer Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make synthesized answers open with a direct 1-2 sentence answer to the question, then the existing complete detail.

**Architecture:** Prompt-only change in `src/agent/synthesizer.py` — rewrite `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` to impose a lead→detail structure and re-scope completeness to the detail section. No code-structure or retrieval change.

**Tech Stack:** Python, pytest. The synthesizer calls `src.generation.llm_client.generate`.

---

### Task 1: Rewrite synthesis prompts to lead with a direct answer

**Files:**
- Modify: `src/agent/synthesizer.py:15-36` (the `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` constants)
- Test: `tests/test_agent/test_synthesizer.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_agent/test_synthesizer.py`)

```python
def test_prompts_instruct_direct_answer_first():
    """Guard the intent: both prompt constants must impose a 'direct answer first,
    then detail' structure so answers don't open with a data dump."""
    from src.agent.synthesizer import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    sys_l = SYSTEM_PROMPT.lower()
    usr_l = USER_PROMPT_TEMPLATE.lower()
    # System prompt sets up the lead->detail structure.
    assert "direct" in sys_l and "answer" in sys_l
    assert "start with" in sys_l          # the lead instruction
    assert "supporting detail" in sys_l or "then" in sys_l
    # User prompt still demands completeness AND a direct lead first.
    assert "first" in usr_l and "direct" in usr_l
    # Completeness guarantees are preserved (don't drop the list-everything behavior).
    assert "every unique item" in usr_l
    assert "deduplicat" in usr_l
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_synthesizer.py::test_prompts_instruct_direct_answer_first -q`
Expected: FAIL (current `SYSTEM_PROMPT` has no "start with"/"direct answer" framing).

- [ ] **Step 3: Replace `SYSTEM_PROMPT`** (lines 15-26)

```python
SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on provided context.

Rules:
- Only answer based on the provided context. Do not use outside knowledge.
- Cite sources by filename, e.g. (2026-01-08_4373866.md). Each context chunk is labeled with its filename.
- If SQL results are provided, reference them in your answer.
- If the context does not contain enough information, say so clearly.

Structure every answer this way:
- START with a direct, 1-2 sentence answer to the exact question asked. Lead with
  the bottom line — the specific value, range, count, or finding — before any
  breakdown. For a list-type question, the opening sentence states what the list
  contains and how many (e.g. "Twelve DHA contracts were awarded in January 2026:").
- THEN provide the supporting detail. Here be THOROUGH and COMPLETE: include ALL
  relevant information from the context, not just the first match; when asked what
  someone said or asked, list EVERY instance; use bullet points or numbered lists,
  and group related items under short headings when it aids clarity.
- Do not restate the same facts in both the opening and the detail.

IMPORTANT: Output ONLY the final answer. Do NOT show your reasoning, self-corrections, internal checks, or thought process. Just provide the clean, organized answer."""
```

- [ ] **Step 4: Replace `USER_PROMPT_TEMPLATE`** (lines 28-36)

```python
USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer using ALL the context above, in two parts:
FIRST, open with a direct 1-2 sentence answer to the exact question — the specific value, range, count, or finding. For a list, state what it contains and how many.
THEN, give the complete supporting detail. Include EVERY unique item from the context — do NOT summarize away or omit ANY entries. If 50 contracts are in the context, list all 50 in the detail. If you run out of space, prioritize listing items over adding descriptions.
DEDUPLICATION: The same item may appear in multiple context sources. Deduplicate by contract number, entity name, or other identifier. If two sources mention the same item, list it ONCE with the most complete details and cite both sources.
Cite sources by filename (e.g. 2026-01-08_4373866.md). Output only the final answer — do NOT include your reasoning process."""
```

- [ ] **Step 5: Run the new test and the full synthesizer suite**

Run: `python -m pytest tests/test_agent/test_synthesizer.py -q`
Expected: PASS (new test passes; existing tests — citations, context build, reasoning-strip — still pass).

- [ ] **Step 6: Commit**

```bash
git add src/agent/synthesizer.py tests/test_agent/test_synthesizer.py
git commit -m "feat: synthesis leads with a direct answer, then detail"
```

---

### Task 2: Deploy and verify live

**Files:** none (verification only)

- [ ] **Step 1: Rebuild + deploy the api**

Run: `docker compose up -d --build api` then wait for `docker compose ps` to show api `(healthy)`.

- [ ] **Step 2: Verify the focused (pay) question leads with a direct answer**

Run: `python examples/query_async.py "what are the pay rates for a gs-13?" --groups executives`
Expected: the answer's FIRST sentence directly states GS-13 pay (a value/range), before any per-year breakdown.

- [ ] **Step 3: Verify the list (contracts) question leads with a summary line**

Run: `python examples/query_async.py "what contracts were awarded by the dha in January?" --groups executives`
Expected: the answer opens with a count/summary sentence (e.g. "N DHA contracts were awarded in January…:"), then the list. (May hit the answer cache; reword slightly to force a fresh run.)

- [ ] **Step 4: Full-suite regression check**

Run: `python -m pytest -q`
Expected: no NEW failures vs the 15-failure baseline (552+ passed).

---

## Self-Review

- **Spec coverage:** SYSTEM_PROMPT revision (Task 1 step 3), USER_PROMPT_TEMPLATE revision (step 4), completeness/dedup preserved (step 4 + test), guard test (step 1), live verification GS-13 + contracts (Task 2), regression check (Task 2 step 4). All spec sections covered.
- **Placeholders:** none — full prompt text and test code inline.
- **Consistency:** test asserts the exact phrases present in the rewritten prompts ("start with", "direct", "every unique item", "deduplicat", "first").
