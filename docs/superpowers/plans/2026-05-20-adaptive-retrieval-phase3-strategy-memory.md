# Adaptive Retrieval Phase 3: Strategy Memory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learn which retrieval strategy works best for each query pattern and override the classifier when historical data shows a better choice, reducing misclassification and improving retrieval efficiency.

**Architecture:** A new `StrategyMemory` model logs strategy performance per query pattern. A new `src/retrieval/strategy_memory.py` module normalizes queries into patterns, logs results, and looks up the historically best strategy. The agent graph checks strategy memory before the classifier and can override it. Pattern normalization replaces specific entities with type placeholders (e.g. "army" → `[ORG]`).

**Tech Stack:** Python, SQLAlchemy, existing LLM classifier, regex-based pattern normalization

---

### Task 1: Add StrategyMemory model and config

**Files:**
- Modify: `src/db/models.py`
- Modify: `src/config.py`

- [ ] **Step 1: Add StrategyMemory model**

In `src/db/models.py`, after the `QueryFeedback` class, add:

```python
class StrategyMemory(Base):
    __tablename__ = "strategy_memory"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query_pattern: Mapped[str] = mapped_column(String, nullable=False, index=True)
    query_type: Mapped[str] = mapped_column(String, default="")
    strategy_used: Mapped[str] = mapped_column(String, default="")
    docs_discovered: Mapped[int] = mapped_column(default=0)
    docs_relevant: Mapped[int] = mapped_column(default=0)
    docs_cited: Mapped[int] = mapped_column(default=0)
    answer_length: Mapped[int] = mapped_column(default=0)
    total_time_seconds: Mapped[float] = mapped_column(default=0.0)
    metadata_fields_useful: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
```

- [ ] **Step 2: Add config setting**

In `src/config.py`, after the PRF settings, add:

```python
    # Strategy memory
    strategy_memory_enabled: bool = True
```

- [ ] **Step 3: Commit**

```bash
git add src/db/models.py src/config.py
git commit -m "feat: add StrategyMemory model and config"
```

---

### Task 2: Create strategy memory module

**Files:**
- Create: `src/retrieval/strategy_memory.py`

- [ ] **Step 1: Create the module**

Create `src/retrieval/strategy_memory.py`:

```python
"""Strategy Memory: learn which strategy works best for each query pattern."""
import logging
import re
from collections import Counter

from sqlalchemy import select

from src.config import settings

logger = logging.getLogger(__name__)

# Common stop words to strip from patterns
_STOP_WORDS = {"what", "which", "who", "how", "when", "where", "the", "a", "an",
               "is", "are", "was", "were", "did", "does", "do", "by", "for",
               "with", "from", "about", "all", "any", "been", "being", "had",
               "has", "have", "its", "that", "this", "those", "can", "could",
               "would", "should", "will", "may", "might", "must", "shall",
               "tell", "me", "list", "show", "find", "give", "get"}

# Patterns for entity type detection
_DATE_PATTERN = re.compile(
    r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|'
    r'aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
    r'[\w.]*\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?',
    re.IGNORECASE
)
_AMOUNT_PATTERN = re.compile(r'\$[\d,.]+\s*(?:million|billion|[kmb])?', re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r'\b\d{4,}\b')


def normalize_query_pattern(query: str) -> str:
    """Normalize a query into a pattern by replacing entities with type placeholders.

    "what contracts did the army award?" → "contracts [ORG] award"
    "what was awarded on Jan 30th?" → "awarded [DATE]"
    "list all contracts above $1 billion" → "contracts above [AMOUNT]"
    """
    text = query.lower().strip().rstrip("?!.")

    # Replace dates
    text = _DATE_PATTERN.sub("[DATE]", text)

    # Replace amounts
    text = _AMOUNT_PATTERN.sub("[AMOUNT]", text)

    # Replace long numbers (contract numbers, IDs)
    text = _NUMBER_PATTERN.sub("[ID]", text)

    # Split into words, remove stop words
    words = text.split()
    filtered = [w for w in words if w not in _STOP_WORDS and w not in ("[DATE]", "[AMOUNT]", "[ID]")]

    # Keep placeholders in their position
    result_words = []
    for w in words:
        if w in ("[DATE]", "[AMOUNT]", "[ID]"):
            if not result_words or result_words[-1] != w:  # dedup adjacent
                result_words.append(w)
        elif w not in _STOP_WORDS:
            result_words.append(w)

    pattern = " ".join(result_words)

    # Replace known org names with [ORG] — use metadata if available
    # For now, replace common known orgs
    for org in ["army", "navy", "air force", "marine corps", "dha", "dla",
                "defense health agency", "defense logistics agency",
                "u.s. army corps of engineers"]:
        pattern = pattern.replace(org, "[ORG]")

    # Collapse whitespace
    pattern = " ".join(pattern.split())
    return pattern


async def log_strategy_result(
    query_text: str,
    query_type: str,
    strategy_used: str,
    docs_discovered: int,
    docs_relevant: int,
    docs_cited: int,
    answer_length: int,
    total_time_seconds: float,
    metadata_fields_useful: list[str] = None,
):
    """Log a strategy result for future pattern matching."""
    if not settings.strategy_memory_enabled:
        return

    from src.db.models import StrategyMemory

    pattern = normalize_query_pattern(query_text)

    try:
        from src.api.routes_ingest import get_metadata_store
        store = get_metadata_store()
        async with store.session_factory() as session:
            session.add(StrategyMemory(
                query_pattern=pattern,
                query_type=query_type,
                strategy_used=strategy_used,
                docs_discovered=docs_discovered,
                docs_relevant=docs_relevant,
                docs_cited=docs_cited,
                answer_length=answer_length,
                total_time_seconds=total_time_seconds,
                metadata_fields_useful=metadata_fields_useful or [],
            ))
            await session.commit()
            logger.info(f"Strategy memory logged: pattern='{pattern}' strategy={strategy_used} "
                       f"relevant={docs_relevant}/{docs_discovered}")
    except Exception as e:
        logger.warning(f"Failed to log strategy memory: {e}")


async def get_best_strategy(query_text: str) -> dict | None:
    """Look up the historically best strategy for a query pattern.

    Returns {"strategy": "sweep", "avg_relevant": 15, "avg_time": 45.0, "count": 5}
    or None if no history exists.
    """
    if not settings.strategy_memory_enabled:
        return None

    from src.db.models import StrategyMemory

    pattern = normalize_query_pattern(query_text)

    try:
        from src.api.routes_ingest import get_metadata_store
        store = get_metadata_store()
        async with store.session_factory() as session:
            result = await session.execute(
                select(StrategyMemory).where(StrategyMemory.query_pattern == pattern)
            )
            records = list(result.scalars().all())
    except Exception:
        return None

    if not records:
        return None

    # Group by strategy, find the one with best precision (relevant/discovered)
    strategy_stats = {}
    for r in records:
        s = r.query_type or r.strategy_used
        if s not in strategy_stats:
            strategy_stats[s] = {"times": [], "relevant": [], "discovered": [], "cited": []}
        strategy_stats[s]["times"].append(r.total_time_seconds)
        strategy_stats[s]["relevant"].append(r.docs_relevant)
        strategy_stats[s]["discovered"].append(r.docs_discovered)
        strategy_stats[s]["cited"].append(r.docs_cited)

    # Pick strategy with highest average precision
    best = None
    best_precision = -1
    for strategy, stats in strategy_stats.items():
        avg_discovered = sum(stats["discovered"]) / len(stats["discovered"])
        avg_relevant = sum(stats["relevant"]) / len(stats["relevant"])
        precision = avg_relevant / avg_discovered if avg_discovered > 0 else 0
        avg_time = sum(stats["times"]) / len(stats["times"])
        count = len(stats["times"])

        if precision > best_precision:
            best_precision = precision
            best = {
                "strategy": strategy,
                "avg_relevant": round(avg_relevant, 1),
                "avg_discovered": round(avg_discovered, 1),
                "avg_time": round(avg_time, 1),
                "precision": round(precision, 3),
                "count": count,
            }

    if best:
        logger.info(f"Strategy memory: pattern='{pattern}' → best={best['strategy']} "
                   f"(precision={best['precision']:.0%} from {best['count']} runs)")

    return best
```

- [ ] **Step 2: Commit**

```bash
git add src/retrieval/strategy_memory.py
git commit -m "feat: create strategy memory module with pattern normalization"
```

---

### Task 3: Log strategy results and check memory at query time

**Files:**
- Modify: `src/admin/routes.py`

- [ ] **Step 1: Log strategy results after queries**

In `src/admin/routes.py`, find the feedback logging block (search for `# Log relevance feedback`). AFTER the entire feedback try/except block, add:

```python
            # Log strategy memory
            try:
                from src.retrieval.strategy_memory import log_strategy_result
                await log_strategy_result(
                    query_text=question,
                    query_type=query_type,
                    strategy_used=query_type,
                    docs_discovered=len({c.metadata.doc_id for c in chunks if c.metadata.doc_id not in {"map-reduce", "knowledge-graph", "metadata-context"}}),
                    docs_relevant=len(mr_relevant_ids) if mr_relevant_ids else 0,
                    docs_cited=len(citations),
                    answer_length=len(answer),
                    total_time_seconds=round(total_time, 2),
                )
            except Exception:
                pass
```

- [ ] **Step 2: Add strategy memory stats to metrics dashboard**

In `src/admin/routes.py`, find the `query_metrics_dashboard` function. Inside the feedback stats block (after the feedback grid HTML), add:

```python
            # Strategy memory stats
            try:
                from src.db.models import StrategyMemory
                sm_result = await session.execute(select(StrategyMemory))
                sm_rows = list(sm_result.scalars().all())
                if sm_rows:
                    patterns = len({r.query_pattern for r in sm_rows})
                    strategies = Counter(r.query_type or r.strategy_used for r in sm_rows)
                    top_strategy = strategies.most_common(1)[0] if strategies else ("—", 0)
                    summary += f"""<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1rem; padding-top:0.5rem; border-top:1px solid #e5e7eb;">
                        <div><strong>{len(sm_rows)}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Strategy Records</span></div>
                        <div><strong>{patterns}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Unique Patterns</span></div>
                        <div><strong>{top_strategy[0]} ({top_strategy[1]}x)</strong><br><span style="font-size:0.8rem; color:#6b7280;">Most Used Strategy</span></div>
                    </div>"""
            except Exception:
                pass
```

Note: `Counter` needs to be imported. Add `from collections import Counter` at the top of the function or use inline import.

- [ ] **Step 3: Commit**

```bash
git add src/admin/routes.py
git commit -m "feat: log strategy results and show strategy stats in dashboard"
```

---

### Task 4: Settings UI + README update

**Files:**
- Modify: `src/admin/templates/settings.html`
- Modify: `src/admin/routes.py`
- Modify: `README.md`

- [ ] **Step 1: Add strategy memory toggle to settings**

In `src/admin/templates/settings.html`, inside the Relevance Feedback section, after the PRF form group, add:

```html
            <div class="form-group">
                <label for="strategy_memory_enabled">Strategy Memory</label>
                <select id="strategy_memory_enabled" name="strategy_memory_enabled">
                    <option value="true" {{ 'selected' if settings.strategy_memory_enabled }}>Enabled</option>
                    <option value="false" {{ 'selected' if not settings.strategy_memory_enabled }}>Disabled</option>
                </select>
                <span style="font-size:0.8rem; color:#6b7280;">Learn which strategy works best for each query pattern</span>
            </div>
```

- [ ] **Step 2: Add to settings save handler**

In `src/admin/routes.py`, add to `save_settings` signature after `prf_enabled`:

```python
    strategy_memory_enabled: bool = Form(True),
```

In-memory update:
```python
    settings.strategy_memory_enabled = strategy_memory_enabled
```

Env lines:
```python
    env_lines["STRATEGY_MEMORY_ENABLED"] = str(settings.strategy_memory_enabled).lower()
```

- [ ] **Step 3: Update README**

In `README.md`, update the Adaptive Retrieval section. After the PRF bullet, add:

```markdown
- **Strategy Memory** -- learns which retrieval strategy (sweep, lookup, etc.) works best for each query pattern. Tracks precision per pattern over time
```

- [ ] **Step 4: Commit and push**

```bash
git add src/admin/templates/settings.html src/admin/routes.py README.md
git commit -m "feat: add strategy memory settings and update README"
git push origin master
```
