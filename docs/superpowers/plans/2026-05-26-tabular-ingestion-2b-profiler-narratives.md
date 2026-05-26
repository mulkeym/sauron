# Tabular Ingestion — Plan 2b: Table Profiler + Deterministic Row Narratives

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce, for a clean table, (a) a one-LLM-call profile that labels columns and tags which are keys vs. measures, and (b) deterministic natural-language narratives for each row that carry real semantic signal for retrieval — without any per-row LLM calls.

**Architecture:** A new `src/ingestion/table_profiler.py` with two independent units: `profile_table` (one LLM call → `TableProfile`, with a heuristic fallback so ingestion never breaks) and `build_row_narratives` (pure templating from the profile's key/measure columns). No pipeline changes, no DB, no embeddings — Plan 2c wires these into ingestion. The LLM call is injected (`generate_fn`) so tests are deterministic.

**Tech Stack:** Python 3.11, the existing `src.generation.llm_client` (`generate`, `parse_json_response`), pytest. Tests run inside the app image: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest <path> -q`.

**Spec:** `docs/superpowers/specs/2026-05-25-tabular-spreadsheet-ingestion-design.md` (Component 2 — profiling + narratives half). **Depends on:** nothing new at runtime (operates on plain Python lists/dicts); conceptually consumes Plan 1's column names/dtypes and feeds Plan 2c.

**Key design decision (from the spec):** row narratives are **deterministic templates**, NOT per-row LLM output — per-row LLM does not scale on the local model and risks fabricating derived values. The LLM is used **once per table** to label columns; narratives then restate only what is literally in each row.

---

## File Structure

- `src/ingestion/table_profiler.py` — **create**. Owns: `TableProfile` (dataclass), `_heuristic_profile` (LLM-free fallback), `profile_table` (one LLM call + parse + validate + fallback), `_fmt_cell`, `row_narrative`, `build_row_narratives`. Single responsibility: turn a clean table's columns+rows into a profile and per-row narratives.
- `tests/test_ingestion/test_table_profiler.py` — **create**.

## The `TableProfile` contract (defined Task 1, used throughout)

```python
@dataclass
class TableProfile:
    column_descriptions: dict[str, str]  # safe_col_name -> human label/description
    key_columns: list[str]               # identifying/dimension columns (safe names)
    measure_columns: list[str]           # numeric value columns (safe names)
    table_description: str               # one sentence about the table
```

Column names everywhere are the **safe names** from Plan 2a's `_safe_column_names` (the caller passes those in). Profiles are validated so `key_columns`/`measure_columns` only ever contain names present in the provided `col_names`.

---

### Task 1: `TableProfile` + `_heuristic_profile`

**Files:**
- Create: `src/ingestion/table_profiler.py`
- Test: `tests/test_ingestion/test_table_profiler.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion/test_table_profiler.py`:

```python
"""Tests for src/ingestion/table_profiler.py — per-table profile + row narratives."""
from src.ingestion.table_profiler import TableProfile, _heuristic_profile


def test_heuristic_profile_splits_keys_and_measures_by_dtype():
    p = _heuristic_profile(
        ["grade", "step", "salary"],
        ["text", "number", "number"],
    )
    assert isinstance(p, TableProfile)
    # text columns are keys; number columns are measures
    assert p.key_columns == ["grade"]
    assert p.measure_columns == ["step", "salary"]
    # descriptions default to the column name itself
    assert p.column_descriptions == {"grade": "grade", "step": "step", "salary": "salary"}
    assert p.table_description  # non-empty


def test_heuristic_profile_all_text_has_no_measures():
    p = _heuristic_profile(["name", "city"], ["text", "text"])
    assert p.key_columns == ["name", "city"]
    assert p.measure_columns == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_table_profiler.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingestion.table_profiler'`.

- [ ] **Step 3: Create the module with `TableProfile` and `_heuristic_profile`**

Create `src/ingestion/table_profiler.py`:

```python
"""Per-table profiling (one LLM call) and deterministic per-row narratives.

``profile_table`` labels a clean table's columns and tags keys vs. measures
with a single LLM call (falling back to a dtype heuristic on any failure).
``build_row_narratives`` then restates each row as a natural-language string
deterministically — no per-row LLM calls — so the embeddings carry semantic
signal while the raw row stays the source of truth.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TableProfile:
    column_descriptions: dict = field(default_factory=dict)  # safe_col_name -> label
    key_columns: list = field(default_factory=list)
    measure_columns: list = field(default_factory=list)
    table_description: str = ""


def _heuristic_profile(col_names: list[str], column_dtypes: list[str]) -> TableProfile:
    """LLM-free profile: number columns are measures, the rest are keys.

    Used as the fallback whenever the LLM profiling call fails or returns
    unusable output, so ingestion never breaks on a profiling error.
    """
    key_columns = [c for c, dt in zip(col_names, column_dtypes) if dt != "number"]
    measure_columns = [c for c, dt in zip(col_names, column_dtypes) if dt == "number"]
    return TableProfile(
        column_descriptions={c: c for c in col_names},
        key_columns=key_columns,
        measure_columns=measure_columns,
        table_description="Table with columns: " + ", ".join(col_names),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_table_profiler.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/table_profiler.py tests/test_ingestion/test_table_profiler.py
git commit -m "feat: TableProfile + heuristic (LLM-free) profile fallback

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `profile_table` (one LLM call + parse + validate + fallback)

**Files:**
- Modify: `src/ingestion/table_profiler.py`
- Test: `tests/test_ingestion/test_table_profiler.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_table_profiler.py` (extend the import to add `profile_table`):

```python
import json

from src.ingestion.table_profiler import profile_table

_COLS = ["grade", "step", "salary"]
_DTYPES = ["text", "number", "number"]
_SAMPLE = [["GS-12", 5, 86415], ["GS-13", 5, 102000]]


def _fake_generate(payload):
    """Return a generate_fn that always yields json.dumps(payload)."""
    def _gen(system_prompt, user_prompt, **kwargs):
        return json.dumps(payload)
    return _gen


def test_profile_table_uses_llm_output():
    gen = _fake_generate({
        "column_descriptions": {"grade": "Pay grade", "step": "Step", "salary": "Annual salary"},
        "key_columns": ["grade", "step"],
        "measure_columns": ["salary"],
        "table_description": "GS pay by grade and step",
    })
    p = profile_table("Pay", _COLS, _DTYPES, _SAMPLE, generate_fn=gen)
    assert p.key_columns == ["grade", "step"]
    assert p.measure_columns == ["salary"]
    assert p.column_descriptions["salary"] == "Annual salary"
    assert p.table_description == "GS pay by grade and step"


def test_profile_table_drops_columns_not_in_table():
    gen = _fake_generate({
        "column_descriptions": {"grade": "Pay grade"},
        "key_columns": ["grade", "bogus"],     # bogus is not a real column
        "measure_columns": ["salary", "also_fake"],
        "table_description": "x",
    })
    p = profile_table("Pay", _COLS, _DTYPES, _SAMPLE, generate_fn=gen)
    assert p.key_columns == ["grade"]          # bogus filtered out
    assert p.measure_columns == ["salary"]     # also_fake filtered out
    # columns the LLM didn't describe still get a default label
    assert p.column_descriptions["step"] == "step"


def test_profile_table_falls_back_when_llm_raises():
    def boom(system_prompt, user_prompt, **kwargs):
        raise RuntimeError("LLM down")
    p = profile_table("Pay", _COLS, _DTYPES, _SAMPLE, generate_fn=boom)
    # heuristic fallback: number cols -> measures
    assert p.key_columns == ["grade"]
    assert p.measure_columns == ["step", "salary"]


def test_profile_table_falls_back_on_unparseable_output():
    def junk(system_prompt, user_prompt, **kwargs):
        return "not json at all"
    p = profile_table("Pay", _COLS, _DTYPES, _SAMPLE, generate_fn=junk)
    assert p.key_columns == ["grade"]
    assert p.measure_columns == ["step", "salary"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_table_profiler.py -k profile_table -q`
Expected: FAIL — `ImportError: cannot import name 'profile_table'`.

- [ ] **Step 3: Implement `profile_table`**

Add to `src/ingestion/table_profiler.py`:

```python
_PROFILE_SYSTEM = (
    "You label spreadsheet columns. Given a table's column names and a few "
    "sample rows, return ONLY a JSON object with these keys:\n"
    '  "column_descriptions": object mapping each column name to a short human label,\n'
    '  "key_columns": array of the column names that identify a row (categories/dimensions),\n'
    '  "measure_columns": array of the numeric value columns,\n'
    '  "table_description": one sentence describing the table.\n'
    "Use ONLY the provided column names. Output JSON only, no prose."
)


def profile_table(sheet_name: str, col_names: list[str], column_dtypes: list[str],
                  sample_rows: list, generate_fn=None) -> TableProfile:
    """Profile a clean table with a single LLM call; fall back to a heuristic.

    ``generate_fn`` defaults to the real LLM client; tests inject a fake. Any
    failure (LLM error, unparseable output, missing keys) yields the dtype
    heuristic so ingestion never breaks. The returned profile's key/measure
    columns are filtered to names actually present in ``col_names``, and every
    column gets a description (defaulting to its own name).
    """
    if generate_fn is None:
        from src.generation.llm_client import generate as generate_fn

    try:
        from src.generation.llm_client import parse_json_response
        sample_text = "\n".join(" | ".join("" if c is None else str(c) for c in row)
                                for row in sample_rows[:5])
        raw = generate_fn(
            system_prompt=_PROFILE_SYSTEM,
            user_prompt=f"Sheet: {sheet_name}\nColumns: {col_names}\nSample rows:\n{sample_text}",
            temperature=0.0,
            max_tokens=1024,
        )
        data = parse_json_response(raw)
        valid = set(col_names)
        key_columns = [c for c in data.get("key_columns", []) if c in valid]
        measure_columns = [c for c in data.get("measure_columns", []) if c in valid]
        raw_desc = data.get("column_descriptions", {}) or {}
        descriptions = {c: str(raw_desc.get(c, c)) for c in col_names}
        table_description = str(data.get("table_description", "")) or (
            "Table with columns: " + ", ".join(col_names)
        )
        if not key_columns and not measure_columns:
            # LLM gave us nothing usable about structure -> heuristic
            return _heuristic_profile(col_names, column_dtypes)
        return TableProfile(
            column_descriptions=descriptions,
            key_columns=key_columns,
            measure_columns=measure_columns,
            table_description=table_description,
        )
    except Exception as e:
        logger.warning(f"Table profiling failed for '{sheet_name}', using heuristic: {e}")
        return _heuristic_profile(col_names, column_dtypes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_table_profiler.py -k profile_table -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/table_profiler.py tests/test_ingestion/test_table_profiler.py
git commit -m "feat: profile_table (one LLM call, heuristic fallback)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `_fmt_cell` + `row_narrative`

**Files:**
- Modify: `src/ingestion/table_profiler.py`
- Test: `tests/test_ingestion/test_table_profiler.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_table_profiler.py` (extend the import to add `_fmt_cell`, `row_narrative`):

```python
from src.ingestion.table_profiler import _fmt_cell, row_narrative

_PROFILE = TableProfile(
    column_descriptions={"grade": "Pay grade", "step": "Step", "salary": "Annual salary"},
    key_columns=["grade", "step"],
    measure_columns=["salary"],
    table_description="GS pay",
)


def test_fmt_cell():
    assert _fmt_cell(None) == "(not specified)"
    assert _fmt_cell("") == "(not specified)"
    assert _fmt_cell("  ") == "(not specified)"
    assert _fmt_cell("GS-12") == "GS-12"
    assert _fmt_cell(5) == "5"


def test_row_narrative_uses_labels_and_keys_then_measures():
    text = row_narrative(["grade", "step", "salary"], _PROFILE, ["GS-12", 5, 86415])
    assert text == "Pay grade=GS-12, Step=5: Annual salary is 86415"


def test_row_narrative_marks_missing_cells():
    # row shorter than the columns -> missing cells are explicit, never fabricated
    text = row_narrative(["grade", "step", "salary"], _PROFILE, ["GS-12"])
    assert text == "Pay grade=GS-12, Step=(not specified): Annual salary is (not specified)"


def test_row_narrative_measures_only():
    profile = TableProfile(column_descriptions={"x": "X"}, key_columns=[], measure_columns=["x"])
    assert row_narrative(["x"], profile, [42]) == "X is 42"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_table_profiler.py -k "fmt_cell or row_narrative" -q`
Expected: FAIL — `ImportError: cannot import name '_fmt_cell'`.

- [ ] **Step 3: Implement `_fmt_cell` and `row_narrative`**

Add to `src/ingestion/table_profiler.py`:

```python
def _fmt_cell(value) -> str:
    """Render a cell for a narrative; missing values are explicit, never faked."""
    if value is None:
        return "(not specified)"
    s = str(value).strip()
    return s if s else "(not specified)"


def row_narrative(col_names: list[str], profile: TableProfile, row: list) -> str:
    """One deterministic sentence for a row: keys as context, then measures.

    Uses the profile's column descriptions as human labels. Cells absent from
    the row (shorter row) render as "(not specified)" — nothing is fabricated.
    """
    index = {name: i for i, name in enumerate(col_names)}

    def cell(name: str) -> str:
        i = index.get(name)
        if i is None or i >= len(row):
            return "(not specified)"
        return _fmt_cell(row[i])

    keys = [f"{profile.column_descriptions.get(k, k)}={cell(k)}" for k in profile.key_columns]
    measures = [f"{profile.column_descriptions.get(m, m)} is {cell(m)}" for m in profile.measure_columns]
    key_str = ", ".join(keys)
    measure_str = "; ".join(measures)
    if key_str and measure_str:
        return f"{key_str}: {measure_str}"
    return key_str or measure_str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_table_profiler.py -k "fmt_cell or row_narrative" -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/table_profiler.py tests/test_ingestion/test_table_profiler.py
git commit -m "feat: deterministic single-row narrative templating

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `build_row_narratives` (rows → list of narratives)

**Files:**
- Modify: `src/ingestion/table_profiler.py`
- Test: `tests/test_ingestion/test_table_profiler.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_table_profiler.py` (extend the import to add `build_row_narratives`):

```python
from src.ingestion.table_profiler import build_row_narratives


def test_build_row_narratives_one_per_row_with_context():
    rows = [["GS-12", 5, 86415], ["GS-13", 5, 102000]]
    out = build_row_narratives(["grade", "step", "salary"], _PROFILE, rows, context="GS pay")
    assert out == [
        "GS pay — Pay grade=GS-12, Step=5: Annual salary is 86415",
        "GS pay — Pay grade=GS-13, Step=5: Annual salary is 102000",
    ]


def test_build_row_narratives_without_context():
    out = build_row_narratives(["grade", "step", "salary"], _PROFILE, [["GS-12", 5, 86415]])
    assert out == ["Pay grade=GS-12, Step=5: Annual salary is 86415"]


def test_build_row_narratives_empty_rows():
    assert build_row_narratives(["grade"], _PROFILE, []) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_table_profiler.py -k build_row_narratives -q`
Expected: FAIL — `ImportError: cannot import name 'build_row_narratives'`.

- [ ] **Step 3: Implement `build_row_narratives`**

Add to `src/ingestion/table_profiler.py`:

```python
def build_row_narratives(col_names: list[str], profile: TableProfile, data_rows: list,
                         context: str = "") -> list[str]:
    """One narrative string per data row, optionally prefixed with ``context``
    (e.g. the table description) so a retrieved narrative is self-describing."""
    prefix = f"{context} — " if context else ""
    return [f"{prefix}{row_narrative(col_names, profile, row)}" for row in data_rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_table_profiler.py -k build_row_narratives -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/table_profiler.py tests/test_ingestion/test_table_profiler.py
git commit -m "feat: build per-row narratives for a table

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run the whole new test file:

```bash
docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_table_profiler.py -q
```
Expected: all PASS — **13 total** (Task 1: 2; Task 2: 4; Task 3: 4; Task 4: 3).

- [ ] Confirm only the two new files changed:

```bash
git diff --stat <first-commit-of-this-plan>^..HEAD
```
Expected: only `src/ingestion/table_profiler.py` and `tests/test_ingestion/test_table_profiler.py`.

## Notes for the implementer

- **`generate_fn` injection** keeps tests free of patching and lets Plan 2c pass the real `generate`. Do NOT call the LLM at import time or in narratives — narratives are pure templating.
- **Fail-open profiling is intentional:** any profiling error returns `_heuristic_profile`, so a single bad LLM response degrades quality (worse labels) but never blocks ingesting a table.
- **Narratives restate only literal cell values** — there is deliberately no arithmetic, no derived metrics (no "up 12% YoY"), because that would fabricate numbers and embed them. This is the core safety property of the deterministic approach.
- **This plan does NOT wire into ingestion.** Plan 2c calls `profile_table` + `build_row_narratives`, embeds the narratives via `embed_texts`, upserts them to the vector store with the raw row carried in metadata, loads rows into DuckDB (Plan 2a), registers + persists the schema, and loads the registry at startup. Keep this plan free of pipeline/DB/embedding imports.
- **Column names are the safe names** from Plan 2a's `_safe_column_names`; Plan 2c passes those (and the matching `column_dtypes` from Plan 1) so the profile's column references line up with the DuckDB table columns.
```
