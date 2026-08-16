import json
import logging
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

import requests

from src.config import settings

logger = logging.getLogger(__name__)

_llm_session_id: ContextVar[str | None] = ContextVar("llm_session_id", default=None)
_llm_agent_id: ContextVar[str | None] = ContextVar("llm_agent_id", default=None)

_SESSION_HEADER_PRECEDENCE = (
    "x-switchyard-session-id",
    "x-session-id",
    "session-id",
    "x-openwebui-chat-id",
)
_AGENT_HEADER_PRECEDENCE = (
    "x-switchyard-agent-id",
    "x-openwebui-user-id",
)


def _header_map(headers) -> dict[str, str]:
    if not headers:
        return {}
    try:
        items = headers.items()
    except Exception:
        return {}
    out = {}
    for key, value in items:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out[str(key).lower()] = text
    return out


def resolve_llm_identity(*, headers=None, agent_id: str | None = None,
                         session_id: str | None = None) -> tuple[str, str | None]:
    """Resolve Switchyard session + agent ids from inbound headers and caller identity."""
    h = _header_map(headers)
    sid = (session_id or "").strip() or None
    if not sid:
        for key in _SESSION_HEADER_PRECEDENCE:
            if h.get(key):
                sid = h[key]
                break
    if not sid:
        sid = str(uuid.uuid4())
    aid = None
    for key in _AGENT_HEADER_PRECEDENCE:
        if h.get(key):
            aid = h[key]
            break
    if not aid:
        aid = (agent_id or "").strip() or None
    return sid, aid


@contextmanager
def llm_session(*, headers=None, agent_id: str | None = None, session_id: str | None = None):
    """Bind session/agent ids for every LLM HTTP call in this task (and to_thread)."""
    sid, aid = resolve_llm_identity(
        headers=headers, agent_id=agent_id, session_id=session_id,
    )
    t_session = _llm_session_id.set(sid)
    t_agent = _llm_agent_id.set(aid)
    try:
        yield sid
    finally:
        # Starlette iterates sync StreamingResponse generators with one
        # next() per thread, each in a fresh copy_context(). reset() then
        # raises ValueError; the bind is already gone with that copy.
        for token, var in ((t_session, _llm_session_id), (t_agent, _llm_agent_id)):
            try:
                var.reset(token)
            except ValueError:
                pass


def outbound_llm_headers() -> dict[str, str]:
    """Headers to attach on a bound LLM call. Empty when no session is active."""
    sid = _llm_session_id.get()
    if not sid:
        return {}
    headers = {
        "x-switchyard-session-id": sid,
        "x-switchyard-request-id": str(uuid.uuid4()),
    }
    aid = _llm_agent_id.get()
    if aid:
        headers["x-switchyard-agent-id"] = aid
    return headers


def _request_headers() -> dict[str, str]:
    headers = outbound_llm_headers()
    if settings.vllm_api_key:
        headers["Authorization"] = f"Bearer {settings.vllm_api_key}"
    return headers


class LLMError(RuntimeError):
    """Base class for LLM call failures."""


class LLMTimeoutError(LLMError):
    """The LLM request exceeded vllm_request_timeout. Deterministic for a given
    payload size — re-running it only wastes another full timeout."""


class LLMConnectionError(LLMError):
    """Could not reach the LLM endpoint. Transient — worth retrying."""


# OpenAI reasoning models (o-series, gpt-5 family) speak a stricter dialect of the
# chat-completions API than vLLM/Gemma or the standard gpt-4* chat models:
#   - the token budget field is `max_completion_tokens`, not `max_tokens`
#   - only the default `temperature` (1) is accepted; any other value 400s
# We also drop `seed` for them: it isn't honoured at the fixed reasoning temperature
# and risks an "unsupported parameter" 400 on some of these models.
_REASONING_MODEL_RE = re.compile(r"^(o\d|gpt-5)", re.IGNORECASE)


def _is_reasoning_model(model: str) -> bool:
    return bool(_REASONING_MODEL_RE.match((model or "").strip()))


def _is_openai_endpoint(base_url: str) -> bool:
    """True for the hosted OpenAI API. `chat_template_kwargs` is a vLLM-only
    extension that OpenAI rejects on every model, so it must never be sent here."""
    return "api.openai.com" in (base_url or "")


def _build_payload(messages: list, model: str, temperature: float, max_tokens: int,
                   *, thinking: bool = False, stream: bool = False) -> dict:
    """Build a chat-completions payload adapted to the target model/endpoint.

    Standard models (gpt-4*, vLLM/Gemma) keep the historical fields. Reasoning
    models get `max_completion_tokens` and no `temperature`/`seed`. The vLLM-only
    `chat_template_kwargs` thinking toggle is only attached for non-OpenAI endpoints.
    """
    payload = {"model": model, "messages": messages}
    if stream:
        payload["stream"] = True

    if _is_reasoning_model(model):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens
        payload["seed"] = settings.llm_seed

    if thinking and not _is_openai_endpoint(settings.vllm_base_url):
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    return payload


def _call_llm(messages: list, model: str, temperature: float, max_tokens: int,
              *, thinking: bool = False) -> str:
    """Call LLM via requests to an OpenAI-compatible endpoint. When ``thinking``
    is set, enable the model's reasoning (chat-template toggle) and raise the
    token budget. If the served template ignores the toggle, generation simply
    proceeds non-thinking — never an error."""
    if thinking:
        max_tokens = settings.sql_thinking_max_tokens
    logger.info(f"LLM call: model={model}, temperature={temperature}, max_tokens={max_tokens}, thinking={thinking}")

    payload = _build_payload(messages, model, temperature, max_tokens, thinking=thinking)

    headers = _request_headers()

    try:
        resp = requests.post(
            f'{settings.vllm_base_url}/chat/completions',
            json=payload,
            headers=headers,
            timeout=settings.vllm_request_timeout,
            verify=settings.ssl_verify,
        )
        resp.raise_for_status()
        response = resp.json()
    except requests.Timeout:
        raise LLMTimeoutError(f"LLM request timed out after {settings.vllm_request_timeout}s")
    except requests.ConnectionError as e:
        raise LLMConnectionError(f"LLM connection failed: {e}")
    except requests.HTTPError as e:
        # Surface the endpoint's response body — for OpenAI a 400 names the exact
        # offending parameter (e.g. "Unsupported parameter: max_tokens"), which the
        # bare HTTPError status line omits.
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            pass
        raise LLMError(f"LLM HTTP error: {e}" + (f"; body: {body}" if body else ""))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from LLM: {e}\nResponse: {resp.text[:500]}")

    if 'error' in response:
        raise RuntimeError(f"LLM error: {response['error']}")

    if 'choices' not in response or not response['choices']:
        raise RuntimeError(f"No choices in LLM response: {resp.text[:200]}")

    message = response['choices'][0]['message']
    content = message.get('content', '').strip() if message.get('content') else ''

    # Fallback 1: reasoning_content field (thinking models like Gemma via llama.cpp)
    if not content:
        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
        if reasoning:
            logger.info(f"Using reasoning field as content fallback, length: {len(reasoning)}")
            content = reasoning.strip()

    # Fallback 2: extract from <think> blocks
    if not content and message.get('content'):
        raw = message['content']
        think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
        if think_match:
            content = think_match.group(1).strip()
            after = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            if after and len(after) > len(content):
                content = after

    # Fallback 3: other field names
    if not content:
        for field in ['text', 'output', 'result']:
            val = message.get(field, '')
            if isinstance(val, str) and val.strip():
                content = val.strip()
                break

    if not content:
        logger.error(f"LLM returned empty content. Keys: {list(message.keys())}")
        logger.error(f"Message: {json.dumps(message, indent=2)[:1000]}")
        raise RuntimeError(f"LLM returned empty content. Message keys: {list(message.keys())}")

    return content


def generate_stream(system_prompt, user_prompt, temperature=0.1, max_tokens=2048,
                    session_id: str | None = None, agent_id: str | None = None):
    """Stream tokens from the LLM. Yields content strings as they arrive.

    ``session_id`` / ``agent_id`` attach Switchyard headers for this POST
    without a ContextVar bind, so SSE threadpool iteration cannot trip
    'Token was created in a different Context' on generator cleanup.
    """
    payload = _build_payload(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        settings.vllm_model_name, temperature, max_tokens, stream=True,
    )

    if session_id:
        headers = {
            "x-switchyard-session-id": session_id,
            "x-switchyard-request-id": str(uuid.uuid4()),
        }
        if agent_id:
            headers["x-switchyard-agent-id"] = agent_id
        if settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {settings.vllm_api_key}"
    else:
        headers = _request_headers()

    resp = requests.post(
        f'{settings.vllm_base_url}/chat/completions',
        json=payload,
        headers=headers,
        stream=True,
        timeout=settings.vllm_request_timeout,
        verify=settings.ssl_verify,
    )
    resp.raise_for_status()

    buffer = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            # Switchyard / OpenAI emit a final usage chunk with choices=[].
            # dict.get("choices", [{}]) still returns [] when the key is present.
            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
            content = delta.get("content") or ""
            if content:
                buffer += content
                # Strip thinking blocks in real-time
                while "<think>" in buffer and "</think>" in buffer:
                    start = buffer.index("<think>")
                    end = buffer.index("</think>") + len("</think>")
                    buffer = buffer[:start] + buffer[end:]
                if "<think>" in buffer and "</think>" not in buffer:
                    continue
                if buffer:
                    yield buffer
                    buffer = ""
        except json.JSONDecodeError:
            continue

    if buffer:
        buffer = re.sub(r"<think>.*?</think>", "", buffer, flags=re.DOTALL).strip()
        if buffer:
            yield buffer


def generate(system_prompt, user_prompt, temperature=0.1, max_tokens=2048, *, thinking=False):
    """Generate text using the LLM. ``thinking`` enables model reasoning for this call."""
    original_content = _call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=settings.vllm_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
    )

    # Strip <think> blocks, preserve content outside them
    content = re.sub(r"<think>.*?</think>", "", original_content, flags=re.DOTALL).strip()

    # If stripping left nothing, extract from the thinking block
    if not content:
        think_match = re.search(r'<think>(.*?)</think>', original_content, re.DOTALL)
        if think_match:
            content = think_match.group(1).strip()

    # Last resort: return original
    if not content:
        content = original_content.strip()

    if not content:
        raise RuntimeError("LLM returned completely empty response")

    return content


def generate_vision(
    system_prompt: str,
    user_prompt: str,
    image_bytes: bytes,
    *,
    mime_type: str = "image/png",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int | None = None,
) -> str:
    """Multimodal chat-completions call (image + text).

    Uses the OpenAI-compatible content-parts format so it works with OpenAI
    vision models and most vLLM multimodal servers. Failures raise LLMError
    subclasses; callers should fail-open for ingest paths.
    """
    import base64

    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{b64}"
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    model = settings.vllm_model_name
    logger.info(
        f"Vision LLM call: model={model}, image_bytes={len(image_bytes)}, max_tokens={max_tokens}"
    )
    payload = _build_payload(messages, model, temperature, max_tokens, thinking=False)

    headers = _request_headers()

    req_timeout = timeout if timeout is not None else settings.vllm_request_timeout
    try:
        resp = requests.post(
            f"{settings.vllm_base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=req_timeout,
            verify=settings.ssl_verify,
        )
        resp.raise_for_status()
        response = resp.json()
    except requests.Timeout:
        raise LLMTimeoutError(f"Vision LLM request timed out after {req_timeout}s")
    except requests.ConnectionError as e:
        raise LLMConnectionError(f"Vision LLM connection failed: {e}")
    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:500] if e.response is not None else ""
        except Exception:
            pass
        raise LLMError(f"Vision LLM HTTP error: {e} {body}") from e

    try:
        original_content = response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Vision LLM returned unexpected payload: {response!r}") from e

    if isinstance(original_content, list):
        # Some servers return content as a list of parts
        parts = []
        for part in original_content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text") or "")
            elif isinstance(part, str):
                parts.append(part)
        original_content = "".join(parts)

    content = re.sub(r"<think>.*?</think>", "", str(original_content), flags=re.DOTALL).strip()
    if not content:
        content = str(original_content).strip()
    if not content:
        raise RuntimeError("Vision LLM returned completely empty response")
    return content


def parse_json_response(text: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences and thinking blocks."""
    if not text:
        raise ValueError("Empty response text")

    original_text = text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if not text:
        think_match = re.search(r'<think>(.*?)</think>', original_text, re.DOTALL)
        if think_match:
            text = think_match.group(1).strip()
        else:
            raise ValueError("No valid content found after stripping thinking blocks")

    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    if not text:
        raise ValueError("No content remaining after stripping formatting")

    return json.loads(text)
