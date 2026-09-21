"""Buffered model HTTP calls with a cancellable whole-response deadline."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import ssl

import aiohttp
import requests


def _tls_context(verify):
    if not verify:
        return False
    bundle = (verify if isinstance(verify, str) else
              os.environ.get('REQUESTS_CA_BUNDLE') or os.environ.get('CURL_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE'))
    return ssl.create_default_context(cafile=bundle or None)


async def _post(url, *, json, headers, timeout, verify):
    # Unlike requests' read timeout, total cancels even when the provider keeps
    # sending whitespace/heartbeats without ever completing its JSON body.
    limits = aiohttp.ClientTimeout(total=timeout, ceil_threshold=float('inf'))
    try:
        async with aiohttp.ClientSession(timeout=limits, trust_env=True) as session:
            async with session.post(url, json=json, headers=headers, ssl=_tls_context(verify)) as response:
                body = await response.read()
                result = requests.Response()
                result.status_code = response.status
                result.headers.update(response.headers)
                result._content = body
                result.encoding = response.charset or 'utf-8'
                result.url = str(response.url)
                result.reason = response.reason
                return result
    except asyncio.TimeoutError as exc:
        raise requests.Timeout('Whole-response model deadline exceeded') from exc
    except aiohttp.ClientPayloadError as exc:
        raise requests.exceptions.ChunkedEncodingError('Incomplete provider response') from exc
    except aiohttp.ClientError as exc:
        raise requests.ConnectionError('Model endpoint connection failed') from exc


def post_json(url, *, json, headers, timeout, verify):
    """Keep the sync generation interface; cancellation closes the HTTP socket.

    Most callers already run in worker threads. Legacy synchronous callers that
    have an active event loop need a separate loop for the bounded request.
    """
    args = dict(json=json, headers=headers, timeout=timeout, verify=verify)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_post(url, **args))
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='model-http') as executor:
        return executor.submit(lambda: asyncio.run(_post(url, **args))).result()
