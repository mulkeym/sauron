"""Verify the configured TLS policy through LightRAG's real SDK transport."""
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import ipaddress
import json
import ssl
import threading

import pytest
from tenacity import stop_after_attempt, wait_none

from src.config import settings
from src import ssl_config
from src.knowledge import graph_rag


@pytest.fixture
def restore_tls_policy(monkeypatch):
    # Production installs global wrappers; restore them so tests don't leak policy.
    import requests
    monkeypatch.setattr(requests.Session, 'request', requests.Session.request)
    for name in ('httpx', 'httpx2'):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        for cls in (module.Client, module.AsyncClient):
            monkeypatch.setattr(cls, '__init__', cls.__init__)
    monkeypatch.setattr(ssl_config, '_ssl_verify_disabled', False)
    monkeypatch.setattr(settings, 'ssl_verify', True)
    # Avoid changing global warning filters in tests.
    monkeypatch.setattr('urllib3.disable_warnings', lambda *a, **k: None)


@pytest.fixture
def local_https(tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName('localhost'), x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / 'cert.pem', tmp_path / 'key.pem'
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = json.dumps({'id': 'synthetic', 'object': 'chat.completion', 'created': 0,
                'model': 'synthetic', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'entity<|#|>Router A<|#|>Artifact<|#|>A router'}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'https://127.0.0.1:{server.server_port}/v1', cert_path, calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def tls_model(monkeypatch, local_https, restore_tls_policy):
    url, cert_path, calls = local_https
    monkeypatch.setattr(settings, 'vllm_base_url', url)
    monkeypatch.setattr(settings, 'vllm_api_key', 'synthetic-not-a-secret')
    monkeypatch.setattr(settings, 'vllm_model_name', 'synthetic')
    monkeypatch.setattr(settings, 'vllm_request_timeout', 3)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy',
                'SSL_CERT_FILE', 'SSL_CERT_DIR'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    monkeypatch.setattr(graph_rag, 'openai_complete_if_cache',
        graph_rag.openai_complete_if_cache.retry_with(stop=stop_after_attempt(1), wait=wait_none()))
    return cert_path, calls


@pytest.mark.asyncio
async def test_ignore_tls_setting_applies_to_lightrag(tls_model, monkeypatch):
    monkeypatch.setattr(settings, 'ssl_verify', False)
    ssl_config.apply_ssl_verify_setting()
    result = await graph_rag._llm_func('Synthetic TLS test')
    assert result.startswith('entity<|#|>Router A')
    assert len(tls_model[1]) == 1


@pytest.mark.asyncio
async def test_verification_rejects_untrusted_certificate(tls_model):
    ssl_config.apply_ssl_verify_setting()
    with pytest.raises(Exception) as failure:
        await graph_rag._llm_func('Synthetic TLS test')
    error = failure.value
    messages = []
    for _ in range(8):
        messages.append(str(error))
        error = error.__cause__ or error.__context__
        if error is None:
            break
    assert 'certificate' in '\n'.join(messages).lower()
    assert tls_model[1] == []


@pytest.mark.asyncio
async def test_reenable_verification_applies_to_new_lightrag_clients(tls_model, monkeypatch):
    monkeypatch.setattr(settings, 'ssl_verify', False)
    ssl_config.apply_ssl_verify_setting()
    await graph_rag._llm_func('Synthetic TLS test')
    monkeypatch.setattr(settings, 'ssl_verify', True)
    ssl_config.apply_ssl_verify_setting()
    with pytest.raises(Exception):
        await graph_rag._llm_func('Synthetic TLS test')
    assert len(tls_model[1]) == 1


@pytest.mark.asyncio
async def test_trusted_private_ca_works_with_verification_enabled(tls_model, monkeypatch):
    monkeypatch.setenv('SSL_CERT_FILE', str(tls_model[0]))
    ssl_config.apply_ssl_verify_setting()
    assert (await graph_rag._llm_func('Synthetic TLS test')).startswith('entity<|#|>')


@pytest.mark.parametrize('module_name', ['httpx', 'httpx2'])
def test_explicit_verification_is_preserved(restore_tls_policy, monkeypatch, module_name):
    module = pytest.importorskip(module_name)
    monkeypatch.setattr(settings, 'ssl_verify', False)
    ssl_config.apply_ssl_verify_setting()
    # An explicit per-client policy must take precedence over the shared default.
    with module.Client(verify=True, trust_env=False) as client:
        assert client._transport._pool._ssl_context.verify_mode == ssl.CERT_REQUIRED


@pytest.mark.asyncio
async def test_actual_sdk_payload_has_graph_limits_and_thinking_disabled(tls_model, monkeypatch):
    monkeypatch.setattr(settings, 'ssl_verify', False)
    monkeypatch.setattr(settings, 'llm_reasoning_adapter', 'vllm_template')
    monkeypatch.setattr(settings, 'llm_answer_thinking', 'enabled')
    ssl_config.apply_ssl_verify_setting()
    await graph_rag._llm_func('Synthetic bounded extraction')
    payload = tls_model[1][0]
    assert payload['max_tokens'] == 4096
    assert payload['chat_template_kwargs'] == {'enable_thinking': False}
    assert 'extra_body' not in payload  # SDK must flatten these provider controls.
