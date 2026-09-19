"""Answer HTML must stay escaped until the browser Markdown sanitizer runs."""
import html
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request


@pytest.mark.asyncio
async def test_cached_playground_answer_and_groups_are_escaped():
    from src.admin import routes
    answer = '## Steps\n\n<img src=x onerror="alert(1)">\n\n```xml\n<route/>\n```'
    group = '<img src=x onerror="alert(2)">'
    store = AsyncMock()
    store.resolve_play_user_groups.return_value = [group]
    decision = SimpleNamespace(query_vector=[], cached={'answer':answer,'citations':[]},
        cache_time=.01, judge_time=.02, judgment={'confidence':1}, accepted=True)
    pending = []
    with patch('src.admin.routes.get_metadata_store', return_value=store), \
         patch('src.admin.routes.asyncio.create_task', side_effect=pending.append), \
         patch('src.agent.profiles.active_snapshot', return_value={}), \
         patch('src.retrieval.query_cache.judged_cache_lookup', AsyncMock(return_value=decision)), \
         patch('src.figures.service.answer_images', AsyncMock(return_value=[])), \
         patch('src.audit.activity.record_query_activity', AsyncMock()), \
         patch('src.retrieval.metrics.QueryMetricsCollector.save', AsyncMock()):
        response = await routes.playground_start(Request({'type':'http','headers':[]}),
            question='Show steps', play_user='network', mode='full', app_id=0, skip_cache='false')
        for coroutine in pending:
            await coroutine
    query_id = json.loads(response.body)['query_id']
    result = routes._playground_jobs.pop(query_id)
    assert result['step'] == 'complete', result
    assert html.escape(answer) in result['result_html']
    assert html.escape(group) in result['result_html']
    assert '<img' not in result['result_html']
