import asyncio

import pytest

from src.admin import routes


@pytest.mark.asyncio
async def test_playground_deadline_stops_spinner_and_cancels_query():
    cancelled = asyncio.Event()
    routes._playground_jobs['deadline-test'] = {'step':'synthesize','active_substep':'waiting','result_html':''}
    async def blocked():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    try:
        await routes._run_playground_with_deadline('deadline-test', blocked(), .01)
        job = routes._playground_jobs['deadline-test']
        assert job['step'] == 'error' and not job['active_substep']
        assert 'overall deadline' in job['error'] and 'synthesize' in job['error']
        assert cancelled.is_set()
    finally:
        routes._playground_jobs.pop('deadline-test')


@pytest.mark.asyncio
async def test_completed_playground_result_is_not_overwritten():
    routes._playground_jobs['deadline-test'] = {'step':'synthesize'}
    async def complete():
        routes._playground_jobs['deadline-test'].update(step='complete', result_html='answer')
    try:
        await routes._run_playground_with_deadline('deadline-test', complete(), 1)
        assert routes._playground_jobs['deadline-test'] == {'step':'complete','result_html':'answer'}
    finally:
        routes._playground_jobs.pop('deadline-test')
