"""Exercise a real HTTP peer: heartbeats must not defeat the total deadline."""
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import pytest
import requests

from src.generation.http_deadline import post_json, _tls_context


@pytest.fixture
def peer():
    disconnected = threading.Event()
    received = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length']))), self.headers.get('Authorization')))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            try:
                if self.path == '/trickle':
                    for _ in range(100):
                        self.wfile.write(b' ')
                        self.wfile.flush()
                        time.sleep(.02)
                self.wfile.write(b'{"answer":"complete"}')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                disconnected.set()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}', disconnected, received
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_heartbeats_do_not_extend_deadline_and_connection_is_closed(peer):
    url, disconnected, received = peer
    started = time.monotonic()
    with pytest.raises(requests.Timeout):
        post_json(url + '/trickle', json={'model':'fixture'}, headers={}, timeout=.15, verify=True)
    assert time.monotonic() - started < 1
    assert disconnected.wait(1), 'The timed-out client must close its socket, not leave an orphan request.'


def test_completed_request_preserves_payload_auth_and_response(peer):
    url, _, received = peer
    response = post_json(url + '/complete', json={'messages':[{'content':'test'}]},
                         headers={'Authorization':'Bearer fixture'}, timeout=1, verify=True)
    response.raise_for_status()
    assert response.json() == {'answer':'complete'}
    assert received == [('/complete', {'messages':[{'content':'test'}]}, 'Bearer fixture')]


@pytest.mark.asyncio
async def test_sync_adapter_works_when_caller_already_has_an_event_loop(peer):
    response = post_json(peer[0] + '/complete', json={}, headers={}, timeout=1, verify=True)
    assert response.json() == {'answer':'complete'}


def test_existing_enterprise_ca_environment_is_honored(monkeypatch):
    from src.generation import http_deadline
    seen = []
    monkeypatch.setenv('REQUESTS_CA_BUNDLE', '/custom/enterprise.pem')
    monkeypatch.setenv('SSL_CERT_FILE', '/other/system.pem')
    monkeypatch.setattr(http_deadline.ssl, 'create_default_context', lambda **kw: seen.append(kw) or 'context')
    assert _tls_context(True) == 'context'
    assert seen == [{'cafile':'/custom/enterprise.pem'}]
    assert _tls_context(False) is False
    assert len(seen) == 1
