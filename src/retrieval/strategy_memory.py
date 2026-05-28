"""Strategy Memory: learn which strategy works best for each query pattern."""
import logging
import re
from collections import Counter

from sqlalchemy import select

from src.config import settings

logger = logging.getLogger(__name__)

_STOP_WORDS = {"what", "which", "who", "how", "when", "where", "the", "a", "an",
               "is", "are", "was", "were", "did", "does", "do", "by", "for",
               "with", "from", "about", "all", "any", "been", "being", "had",
               "has", "have", "its", "that", "this", "those", "can", "could",
               "would", "should", "will", "may", "might", "must", "shall",
               "tell", "me", "list", "show", "find", "give", "get"}

_DATE_PATTERN = re.compile(
    r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|'
    r'aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
    r'[\w.]*\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?',
    re.IGNORECASE
)
_AMOUNT_PATTERN = re.compile(r'\$[\d,.]+\s*(?:million|billion|[kmb])?', re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r'\b\d{4,}\b')


def normalize_query_pattern(query: str) -> str:
    text = query.lower().strip().rstrip("?!.")
    text = _DATE_PATTERN.sub("[DATE]", text)
    text = _AMOUNT_PATTERN.sub("[AMOUNT]", text)
    text = _NUMBER_PATTERN.sub("[ID]", text)

    words = text.split()
    result_words = []
    for w in words:
        if w in ("[DATE]", "[AMOUNT]", "[ID]"):
            if not result_words or result_words[-1] != w:
                result_words.append(w)
        elif w not in _STOP_WORDS:
            result_words.append(w)

    pattern = " ".join(result_words)
    for org in ["army", "navy", "air force", "marine corps", "dha", "dla",
                "defense health agency", "defense logistics agency",
                "u.s. army corps of engineers"]:
        pattern = pattern.replace(org, "[ORG]")

    return " ".join(pattern.split())


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

    strategy_stats = {}
    for r in records:
        s = r.query_type or r.strategy_used
        if s not in strategy_stats:
            strategy_stats[s] = {"times": [], "relevant": [], "discovered": [], "cited": []}
        strategy_stats[s]["times"].append(r.total_time_seconds)
        strategy_stats[s]["relevant"].append(r.docs_relevant)
        strategy_stats[s]["discovered"].append(r.docs_discovered)
        strategy_stats[s]["cited"].append(r.docs_cited)

    # Composite ranking: primary = avg docs cited (truest success proxy),
    # tiebreak = avg relevant. Precision is reported but no longer the key
    # (it structurally favored narrow strategies and punished recall).
    def composite(stats):
        avg_cited = sum(stats["cited"]) / len(stats["cited"])
        avg_relevant = sum(stats["relevant"]) / len(stats["relevant"])
        return avg_cited + 0.001 * avg_relevant  # relevant breaks cited ties

    ranked = sorted(
        strategy_stats.items(), key=lambda kv: composite(kv[1]), reverse=True
    )
    top_strategy, top_stats = ranked[0]
    top_comp = composite(top_stats)
    runner_comp = composite(ranked[1][1]) if len(ranked) > 1 else 0.0
    margin = (top_comp - runner_comp) / top_comp if top_comp > 0 else (1.0 if len(ranked) == 1 else 0.0)

    avg_discovered = sum(top_stats["discovered"]) / len(top_stats["discovered"])
    avg_relevant = sum(top_stats["relevant"]) / len(top_stats["relevant"])
    avg_cited = sum(top_stats["cited"]) / len(top_stats["cited"])
    precision = avg_relevant / avg_discovered if avg_discovered > 0 else 0
    best = {
        "strategy": top_strategy,
        "avg_relevant": round(avg_relevant, 1),
        "avg_discovered": round(avg_discovered, 1),
        "avg_cited": round(avg_cited, 1),
        "avg_time": round(sum(top_stats["times"]) / len(top_stats["times"]), 1),
        "precision": round(precision, 3),
        "count": len(top_stats["times"]),
        "margin": round(margin, 3),
    }

    if best:
        logger.info(f"Strategy memory: pattern='{pattern}' -> best={best['strategy']} "
                   f"(cited={best['avg_cited']}, margin={best['margin']:.0%}, n={best['count']})")

    return best
