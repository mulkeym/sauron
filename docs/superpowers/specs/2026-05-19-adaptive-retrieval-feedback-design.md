# Adaptive Retrieval with Relevance Feedback

## Overview

Build a learning retrieval system that improves over time by logging which documents and chunks were useful for each query, then leveraging that history to boost retrieval accuracy for future similar queries. No model training required — uses table-based feedback signals that compound over time and never invalidate when new documents are added.

## Problem

Current retrieval is stateless — every query starts fresh. If someone asks "what contracts did the army award?" on Monday and gets great results from 20 specific documents, then asks the same question on Tuesday, the system rediscovers those 20 documents from scratch. There's no memory of what worked.

The query cache partially solves this by storing full answers, but it goes stale when new documents are added. We need something that learns relevance patterns without caching answers.

## Solution: Three-Phase Adaptive Retrieval

### Phase 1: Relevance Feedback Log

**What:** Log which documents were actually useful for each query. Use that history to boost document ranking for similar future queries.

#### New Table: `query_feedback`

```python
class QueryFeedback(Base):
    __tablename__ = "query_feedback"
    id: int (PK)
    query_hash: str           # SHA-256 of normalized query text
    query_text: str           # original question
    query_vector: bytes       # serialized embedding vector (for similarity search)
    query_type: str           # sweep, lookup, analytical, etc.
    doc_id: str               # document that was used
    filename: str             # for quick display
    relevance_score: float    # retrieval score
    was_cited: bool           # appeared in final answer citations
    was_in_map_reduce: bool   # was MAP'd and returned relevant data (not NO_RELEVANT_DATA)
    user_groups: list[str]    # ACL context
    response_quality: str     # "good", "partial", "irrelevant" (future: user thumbs up/down)
    created_at: datetime
```

#### Logging (automatic, after every query)

After the synthesizer generates an answer:
1. For each cited document → log with `was_cited=True`
2. For each MAP'd document that returned data → log with `was_in_map_reduce=True`
3. For each MAP'd document that returned NO_RELEVANT_DATA → log with `was_in_map_reduce=False`
4. Store the query vector for future similarity matching

#### Retrieval Boost (automatic, at query time)

During document discovery (in map_reduce.py and sweep.py):
1. Embed the current query
2. Find past queries with cosine similarity > 0.85 from `query_feedback`
3. Get the doc_ids that were cited/useful for those past queries
4. Boost those doc_ids in the current candidate scoring:
   - `was_cited=True` → +0.3 boost
   - `was_in_map_reduce=True` → +0.2 boost
   - `was_in_map_reduce=False` → -0.1 penalty (was tried but irrelevant)

This means: "last time someone asked about army contracts, these 20 docs were useful → boost them for this similar query." New documents still get discovered normally through vector/metadata search — they just start with no historical signal.

#### Decay

Feedback older than 90 days gets reduced weight (multiply by 0.5 per 90 days). This prevents stale patterns from dominating as the document corpus evolves.

### Phase 2: Pseudo-Relevance Feedback (PRF)

**What:** Use the top results from an initial fast search to expand and refine the query, then search again. Works immediately — no history needed.

#### How It Works

1. Initial search: vector search for the query → top 5 results
2. Extract key terms from those top 5 documents' `metadata_tags`:
   - Entities, organizations, identifiers, topics
   - Weight terms by frequency across the top 5
3. Expand the query: append the top extracted terms to the original question
4. Search again with the expanded query → better recall

#### Example

```
Original query: "what contracts did the army award?"
Top 5 results mention: "U.S. Army Corps of Engineers", "firm-fixed-price", 
                        "W912" contract prefixes, "construction", "defense"

Expanded query: "what contracts did the army award? U.S. Army Corps of Engineers 
                 firm-fixed-price W912 construction defense"

Second search: finds more contract documents that use these specific terms
```

#### When To Use

PRF runs automatically on SWEEP queries where the initial search returns fewer than the expected number of relevant documents. Not needed for LOOKUP (already specific enough).

### Phase 3: Strategy Memory

**What:** Learn which retrieval strategy and parameters work best for each type of question. Simple frequency counting, no ML.

#### New Table: `strategy_memory`

```python
class StrategyMemory(Base):
    __tablename__ = "strategy_memory"
    id: int (PK)
    query_pattern: str        # normalized pattern (e.g. "contracts awarded by [ORG]")
    query_type: str           # sweep, lookup, etc.
    strategy_used: str        # map_reduce, sweep, lookup, cross_reference
    docs_discovered: int      # how many docs were found
    docs_relevant: int        # how many had useful data
    docs_cited: int           # how many appeared in answer
    answer_length: int        # chars in final answer
    total_time_seconds: float # wall clock time
    metadata_fields_useful: list[str]  # which metadata fields matched (organizations, amounts, etc.)
    created_at: datetime
```

#### How It Works

After each query completes:
1. Normalize the query into a pattern: "what contracts did the army award?" → "contracts [ORG] award"
2. Log the strategy used, how many docs were relevant, and which metadata fields were useful
3. Over time, patterns emerge: "contracts [ORG] award" → sweep works best, organizations field most useful, typically 15-30 docs relevant

On future queries:
1. Classify the query into a pattern
2. Look up the best-performing strategy for that pattern
3. If the classifier disagrees, prefer the historically-proven strategy
4. Pre-prioritize the metadata fields that were useful for this pattern

#### Pattern Normalization

Replace specific entities with type placeholders:
- "army" → `[ORG]`, "Jan 30th" → `[DATE]`, "$1 billion" → `[AMOUNT]`
- "what contracts did the army award?" → "contracts [ORG] award"
- "what was awarded on Jan 30th?" → "awarded [DATE]"
- Use the query's own metadata extraction (entities, dates, amounts) for replacement

## Data Flow

```
Query arrives
  │
  ├─ Phase 3: Check strategy memory for similar patterns
  │  → Override classifier if historical strategy was better
  │
  ├─ Phase 1: Check relevance feedback for similar past queries
  │  → Boost previously-useful doc_ids in candidate scoring
  │
  ├─ Normal retrieval (vector search + metadata scan)
  │
  ├─ Phase 2: If SWEEP and initial results sparse, run PRF
  │  → Expand query with terms from top results, search again
  │
  ├─ MAP / Synthesize as normal
  │
  └─ After answer generated:
     ├─ Phase 1: Log (query, doc_id, was_cited, was_in_map_reduce) to query_feedback
     └─ Phase 3: Log (pattern, strategy, doc counts, time) to strategy_memory
```

## Admin UI

### Feedback Dashboard (new section in Settings or standalone page)

- **Query History**: list of recent queries with doc counts, cited counts, strategy used
- **Top Relevant Documents**: documents ranked by how often they're cited across queries
- **Strategy Performance**: which strategies work best for which query patterns
- **Feedback Stats**: total logged queries, unique patterns, average docs per query

### Manual Feedback (future)

- Thumbs up/down on playground answers
- Updates `response_quality` in `query_feedback`
- Improves boost weights: thumbs-up docs get higher boost, thumbs-down get penalty

## Configuration

```python
# config.py
feedback_enabled: bool = True
feedback_similarity_threshold: float = 0.85  # cosine sim for matching past queries
feedback_boost_cited: float = 0.3            # boost for previously-cited docs
feedback_boost_relevant: float = 0.2         # boost for MAP-relevant docs
feedback_penalty_irrelevant: float = 0.1     # penalty for MAP-irrelevant docs
feedback_decay_days: int = 90                # halve weight after this many days
prf_enabled: bool = True                     # enable pseudo-relevance feedback
prf_min_results: int = 5                     # trigger PRF if fewer than this many docs found
strategy_memory_enabled: bool = True
```

## Performance Impact

**Storage:** ~1KB per query-document pair. 100 queries/day × 20 docs/query = 2KB/day. Negligible.

**Query time:** +1 DB query to find similar past queries (~5ms). +1 vector similarity search on feedback vectors (~10ms). Total: ~15ms overhead. Negligible compared to LLM calls.

**Ingestion:** No impact. Feedback is query-side only.

## Key Properties

- **Never invalidates** — old feedback stays valid, new docs discovered normally
- **Compounds over time** — more queries = more accurate retrieval
- **No model training** — pure table-based frequency counting and vector similarity
- **Per-user-group** — different ACL groups build different relevance profiles
- **Graceful degradation** — if feedback tables are empty, system works exactly as before

## Migration for Existing System

- New tables created on startup via SQLAlchemy `create_all`
- No changes to existing data
- System works without feedback (returns to current behavior)
- Feedback accumulates naturally as users query

## Files Created/Modified

### Phase 1
- Create: `src/retrieval/feedback.py` — logging + boost lookup logic
- Modify: `src/db/models.py` — add QueryFeedback model
- Modify: `src/db/metadata.py` — add feedback CRUD methods
- Modify: `src/agent/synthesizer.py` — log feedback after answer generation
- Modify: `src/agent/strategies/map_reduce.py` — apply relevance boost during discovery
- Modify: `src/agent/strategies/sweep.py` — apply relevance boost during discovery

### Phase 2
- Create: `src/retrieval/prf.py` — pseudo-relevance feedback query expansion
- Modify: `src/agent/strategies/map_reduce.py` — trigger PRF when results sparse
- Modify: `src/agent/strategies/sweep.py` — trigger PRF when results sparse

### Phase 3
- Modify: `src/db/models.py` — add StrategyMemory model
- Create: `src/retrieval/strategy_memory.py` — pattern normalization, strategy logging, lookup
- Modify: `src/agent/graph.py` — check strategy memory before retrieval, log after
- Modify: `src/admin/routes.py` — feedback dashboard
- Modify: `src/admin/templates/` — new dashboard page or settings section
- Modify: `src/config.py` — feedback configuration settings
