"""Query performance metrics collection and reporting."""
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QueryMetricsCollector:
    """Collects metrics during a query execution. Create one per query."""
    query_text: str = ""
    query_type: str = ""
    strategy_used: str = ""
    user_groups: list = field(default_factory=list)

    # Document counts
    docs_discovered: int = 0
    docs_map_read: int = 0
    docs_relevant: int = 0  # MAP returned data (not NO_RELEVANT_DATA)
    docs_cited: int = 0

    # Timing
    _start_time: float = 0.0
    _retrieval_start: float = 0.0
    retrieval_time: float = 0.0
    map_time: float = 0.0
    synthesis_time: float = 0.0
    total_time_seconds: float = 0.0

    # Context
    context_chars: int = 0
    answer_length: int = 0

    # Future feedback system
    feedback_boost_applied: int = 0
    prf_triggered: bool = False
    cache_hit: bool = False

    def start(self):
        self._start_time = time.time()

    def finish(self):
        self.total_time_seconds = round(time.time() - self._start_time, 2)
        if self.docs_map_read > 0:
            self.map_precision = round(self.docs_relevant / self.docs_map_read, 3)
        else:
            self.map_precision = 0.0

    @property
    def map_precision(self) -> float:
        if self.docs_map_read > 0:
            return round(self.docs_relevant / self.docs_map_read, 3)
        return 0.0

    @map_precision.setter
    def map_precision(self, value):
        pass  # computed property, ignore sets

    async def save(self):
        """Persist metrics to database."""
        try:
            from src.api.routes_ingest import get_metadata_store
            store = get_metadata_store()
            from src.db.models import QueryMetrics
            async with store.session_factory() as session:
                record = QueryMetrics(
                    query_text=self.query_text[:500],
                    query_type=self.query_type,
                    strategy_used=self.strategy_used,
                    user_groups=self.user_groups,
                    docs_discovered=self.docs_discovered,
                    docs_map_read=self.docs_map_read,
                    docs_relevant=self.docs_relevant,
                    docs_cited=self.docs_cited,
                    map_precision=self.map_precision,
                    total_time_seconds=self.total_time_seconds,
                    retrieval_time=self.retrieval_time,
                    map_time=self.map_time,
                    synthesis_time=self.synthesis_time,
                    context_chars=self.context_chars,
                    answer_length=self.answer_length,
                    feedback_boost_applied=self.feedback_boost_applied,
                    prf_triggered=self.prf_triggered,
                    cache_hit=self.cache_hit,
                )
                session.add(record)
                await session.commit()
        except Exception as e:
            logger.warning(f"Failed to save query metrics: {e}")

    def log_summary(self):
        """Log a one-line summary."""
        precision = f"{self.map_precision:.0%}" if self.docs_map_read > 0 else "N/A"
        logger.info(
            f"Query metrics: type={self.query_type} strategy={self.strategy_used} "
            f"discovered={self.docs_discovered} map_read={self.docs_map_read} "
            f"relevant={self.docs_relevant} cited={self.docs_cited} "
            f"precision={precision} time={self.total_time_seconds}s "
            f"context={self.context_chars:,}chars answer={self.answer_length}chars"
        )
