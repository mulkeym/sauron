"""Provider request controls and final-channel filtering; no prompt-based guesses."""
from __future__ import annotations

import time
from urllib.parse import urlsplit

import requests

from src.config import settings


class ReasoningConfigurationError(ValueError):
    pass


_cache = {}


def is_openrouter(base_url):
    url = urlsplit(base_url)
    return url.scheme == 'https' and url.hostname == 'openrouter.ai' and url.path.rstrip('/') == '/api/v1'


def reasoning_capability(base_url=None, model=None, adapter=None):
    base_url = (base_url or settings.vllm_base_url).rstrip('/')
    model = model or settings.vllm_model_name
    adapter = adapter or settings.llm_reasoning_adapter
    if is_openrouter(base_url):
        key = (base_url, model, settings.ssl_verify)
        cached = _cache.get(key)
        if cached and cached[0] > time.monotonic():
            return dict(cached[1])
        try:
            response = requests.get(base_url + '/models', timeout=10, verify=settings.ssl_verify)
            response.raise_for_status()
            record = next((m for m in response.json()['data'] if m.get('id') == model), {})
            spec = record.get('reasoning') or {}
            result = {'adapter': 'openrouter', 'supported': 'reasoning' in record.get('supported_parameters', []),
                      'mandatory': bool(spec.get('mandatory')), 'default_enabled': spec.get('default_enabled'),
                      'detail': ('Thinking is supported by OpenRouter model metadata.' if 'reasoning' in record.get('supported_parameters', [])
                                 else 'This selected model does not advertise reasoning support in OpenRouter metadata.')}
        except (requests.RequestException, ValueError, KeyError, TypeError):
            raise ReasoningConfigurationError('Cannot verify OpenRouter reasoning support. Retry the capability check or use Provider default.') from None
        _cache[key] = (time.monotonic() + 300, result)
        return dict(result)
    # Explicit operator selection is necessary: an OpenAI-compatible URL alone
    # cannot establish support for vLLM template extensions.
    if adapter == 'vllm_template' and urlsplit(base_url).hostname not in {'api.openai.com', 'openrouter.ai', 'generativelanguage.googleapis.com'}:
        return {'adapter': 'vllm_template', 'supported': True, 'mandatory': False, 'default_enabled': None,
                'detail': 'Operator-selected vLLM template control. Server/template support is not automatically verified.'}
    return {'adapter': 'unsupported', 'supported': False, 'mandatory': False, 'default_enabled': None,
            'detail': 'No verified request control for this endpoint. Use Provider default, or select vLLM only for a compatible server.'}


def reasoning_parameters(enabled, *, model):
    capability = reasoning_capability(model=model)
    if not capability['supported']:
        raise ReasoningConfigurationError(capability['detail'])
    if not enabled and capability['mandatory']:
        raise ReasoningConfigurationError('This model requires reasoning and cannot disable it. Use Provider default or Enabled.')
    if capability['adapter'] == 'openrouter':
        return {'reasoning': {'enabled': enabled, 'exclude': True}, 'provider': {'require_parameters': True}}
    return {'chat_template_kwargs': {'enable_thinking': enabled}}


def answer_reasoning_kwargs():
    mode = settings.llm_answer_thinking
    return {} if mode == 'default' else {'reasoning_mode': mode}


def answer_generation_kwargs():
    """Read live answer controls together; unrelated generation keeps its defaults."""
    return {"temperature": settings.llm_answer_temperature, **answer_reasoning_kwargs()}


class FinalTextFilter:
    """Remove tagged thought channels even when delimiters span SSE chunks.

    Hold ambiguous delimiter prefixes and discard unclosed reasoning at EOF.
    Separate reasoning fields are never passed into this filter.
    """
    pairs = {'<think>': '</think>', '<|channel>thought': '<channel|>'}

    def __init__(self):
        self.buffer = ''
        self.closing = None

    def feed(self, text, *, final=False):
        self.buffer += text
        output = []
        while self.buffer:
            if self.closing:
                pos = self.buffer.find(self.closing)
                if pos < 0:
                    # Keep only a possible split closing delimiter, not thoughts.
                    self.buffer = self.buffer[-(len(self.closing) - 1):] if not final else ''
                    break
                self.buffer = self.buffer[pos + len(self.closing):]
                self.closing = None
                continue
            found = [(self.buffer.find(start), start) for start in self.pairs if start in self.buffer]
            if found:
                pos, start = min(found)
                output.append(self.buffer[:pos])
                self.buffer = self.buffer[pos + len(start):]
                self.closing = self.pairs[start]
                continue
            hold = max((n for start in self.pairs for n in range(1, len(start))
                        if self.buffer.endswith(start[:n])), default=0)
            output.append(self.buffer[:-hold] if hold else self.buffer)
            self.buffer = self.buffer[-hold:] if hold and not final else ''
            break
        return ''.join(output)


def final_text(content):
    if isinstance(content, list):
        content = ''.join(part.get('text', '') for part in content
                          if isinstance(part, dict) and part.get('type') == 'text')
    if not isinstance(content, str):
        return ''
    return FinalTextFilter().feed(content, final=True).strip()
