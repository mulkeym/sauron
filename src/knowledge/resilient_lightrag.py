"""Chunk failure isolation for the pinned LightRAG 1.5.7 pipeline.

Use the upstream parser, cache and merge; only change chunk scheduling. Never
return an empty successful extraction for a failed model call.
"""
import asyncio
import logging

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.exceptions import PipelineCancelledException, IndexFlushError

logger = logging.getLogger(__name__)
FAILURES_KEY = "sauron_failed_chunks"


class PartialGraphError(RuntimeError):
    """At least one chunk failed; the document must not appear complete."""


class ResilientLightRAG(LightRAG):
    async def _process_extract_entities(self, chunk, pipeline_status=None,
                                        pipeline_status_lock=None, truncation_tally=None):
        # A bounded worker pool avoids creating one task per document chunk.
        if not hasattr(self, "_sauron_failures"):
            self._sauron_failures = {}
        pending = iter(chunk.items())
        results = {}
        failures = {}

        async def worker():
            for chunk_id, value in pending:
                try:
                    results[chunk_id] = await super(ResilientLightRAG, self)._process_extract_entities(
                        {chunk_id: value}, pipeline_status, pipeline_status_lock,
                        truncation_tally=truncation_tally,
                    )
                except (asyncio.CancelledError, PipelineCancelledException, IndexFlushError):
                    raise
                except Exception as exc:
                    # Storage errors must abort: continuing after a failed cache
                    # write could create graph state that cannot be recovered.
                    if isinstance(exc, OSError) and not isinstance(exc, TimeoutError):
                        raise
                    failures[chunk_id] = f"{type(exc).__name__}: {exc}"[:500]
                    logger.warning("KG chunk %s failed; continuing other chunks: %s",
                                   chunk_id, failures[chunk_id])
                message = (f"Knowledge graph chunks attempted {len(results) + len(failures)}/{len(chunk)} "
                           f"({len(failures)} failed)")
                logger.info(message)
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = message

        workers = [asyncio.create_task(worker()) for _ in range(
            min(len(chunk), max(1, self.llm_model_max_async)))]
        try:
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            # Persist failure details after ainsert finishes, including when a
            # later merge fails. Each failure remains associated with its source.
            for chunk_id, error in failures.items():
                doc_id = chunk[chunk_id]["full_doc_id"]
                self._sauron_failures.setdefault(doc_id, {})[chunk_id] = error
        if failures and not results:
            raise PartialGraphError(
                f"All {len(chunk)} knowledge graph chunks failed; "
                f"first error: {next(iter(failures.values()))}")
        return [result for chunk_id in chunk for result in results.get(chunk_id, [])]

    async def ainsert(self, *args, **kwargs):
        self._sauron_failures = {}
        messages = []
        try:
            result = await super().ainsert(*args, **kwargs)
        finally:
            for doc_id, failures in self._sauron_failures.items():
                row = await self.doc_status.get_by_id(doc_id)
                if row is None:
                    messages.append(f"Knowledge graph incomplete: missing status for {doc_id}")
                    continue
                total = row.get("chunks_count", len(row.get("chunks_list") or []))
                outcome = ("Successful chunk results are retained; retry to finish. "
                           if row.get("status") == DocStatus.PROCESSED else
                           f"Graph processing did not finish: {row.get('error_msg') or 'unknown error'}. ")
                message = (f"Knowledge graph incomplete: {len(failures)} of {total} chunks failed. "
                           + outcome +
                           f"First failure: {next(iter(failures))}: {next(iter(failures.values()))}")
                messages.append(message)
                row.update(status=DocStatus.FAILED, error_msg=message)
                row["metadata"] = {**(row.get("metadata") or {}), FAILURES_KEY: failures}
                await self.doc_status.upsert({doc_id: row})
            if self._sauron_failures:
                await self.doc_status.index_done_callback()
        if self._sauron_failures:
            raise PartialGraphError("; ".join(messages))
        return result
