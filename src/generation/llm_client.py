import json
import logging
import re
import uuid
import time
from contextlib import contextmanager
from contextvars import ContextVar

import requests

from src.config import settings
from src.generation.http_deadline import post_json

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
    """The buffered LLM request exceeded its whole-response deadline.
    A deadline failure is not automatically retried."""


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
    from urllib.parse import urlsplit
    return urlsplit(base_url or "").hostname == "api.openai.com"


def _build_payload(messages: list, model: str, temperature: float, max_tokens: int,
                   *, thinking: bool = False, stream: bool = False, reasoning_mode: str | None = None) -> dict:
    """Build a chat-completions payload adapted to the target model/endpoint.

    Standard models (gpt-4*, vLLM/Gemma) keep the historical fields. Reasoning
    models get `max_completion_tokens` and no `temperature`/`seed`. Explicit reasoning uses verified OpenRouter metadata or an operator-selected
    vLLM template adapter. Provider-default requests omit reasoning controls.
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

    from src.generation.reasoning import reasoning_parameters, ReasoningConfigurationError
    if reasoning_mode not in {None, "default", "enabled", "disabled"}:
        raise LLMError("Invalid reasoning mode")
    if reasoning_mode in {"enabled", "disabled"} or thinking:
        try:
            payload.update(reasoning_parameters(reasoning_mode != "disabled", model=model))
        except ReasoningConfigurationError as exc:
            if reasoning_mode in {"enabled", "disabled"}:
                raise LLMError(str(exc)) from exc
            # The historical SQL hint was best effort. Do not invent a request
            # extension or break SQL on endpoints with no verified support.
            logger.warning("SQL thinking hint omitted: %s", exc)

    return payload


def _call_llm(messages: list, model: str, temperature: float, max_tokens: int,
              *, thinking: bool = False, reasoning_mode: str | None = None) -> str:
    """Call the configured endpoint. ``thinking`` is the legacy SQL hint;
    explicit answer ``reasoning_mode`` uses the caller's unchanged token budget."""
    if thinking:
        max_tokens = settings.sql_thinking_max_tokens
    logger.info(f"LLM call: model={model}, temperature={temperature}, max_tokens={max_tokens}, thinking={thinking}, answer_reasoning={reasoning_mode or 'default'}")

    payload = _build_payload(messages, model, temperature, max_tokens, thinking=thinking, reasoning_mode=reasoning_mode)

    headers = _request_headers()

    started = time.monotonic()
    deadline = started + settings.vllm_request_timeout
    call_id = uuid.uuid4().hex[:10]
    logger.info("Model request %s started; total deadline=%ss", call_id, settings.vllm_request_timeout)
    try:
        for attempt in range(2):
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise requests.Timeout()
                resp = post_json(
                    f'{settings.vllm_base_url}/chat/completions',
                    json=payload,
                    headers=headers,
                    timeout=remaining,
                    verify=settings.ssl_verify,
                )
                break
            except requests.exceptions.ChunkedEncodingError as e:
                # The provider closed an incomplete HTTP response. Discard it;
                # retry the same request once, never parse partial model output.
                if attempt:
                    raise LLMConnectionError("LLM response ended prematurely after one transport retry") from e
                logger.warning("LLM response ended prematurely; retrying transport once")
        resp.raise_for_status()
        response = resp.json()
    except requests.Timeout:
        logger.warning("Model request %s timed out after %.1fs", call_id, time.monotonic() - started)
        raise LLMTimeoutError(f"Model generation exceeded the {settings.vllm_request_timeout}-second total deadline. The provider did not complete its response. Retry, turn off answer thinking, or increase the model timeout.")
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
        raise LLMError("LLM endpoint returned invalid JSON.") from e

    usage = response.get("usage") or {}
    logger.info("Model request %s completed in %.1fs; provider=%s; completion_tokens=%s; reasoning_tokens=%s",
                call_id, time.monotonic() - started, response.get("provider", "unknown"), usage.get("completion_tokens"),
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))

    if 'error' in response:
        raise RuntimeError(f"LLM error: {response['error']}")

    if 'choices' not in response or not response['choices']:
        raise LLMError("LLM endpoint returned no choices.")

    from src.generation.reasoning import final_text
    choice = response['choices'][0]
    if choice.get('finish_reason') == 'length':
        raise LLMError('LLM output token budget exhausted before completion; increase the answer output limit.')
    content = final_text(choice.get('message', {}).get('content'))
    if not content:
        raise LLMError('LLM returned no final answer; reasoning text is not an answer.')
    return content


def generate_stream(system_prompt, user_prompt, temperature=0.1, max_tokens=2048,
                    session_id: str | None = None, agent_id: str | None = None, *, reasoning_mode: str | None = None):
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
        settings.vllm_model_name, temperature, max_tokens, stream=True, reasoning_mode=reasoning_mode,
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

    from src.generation.reasoning import FinalTextFilter
    parser = FinalTextFilter()
    emitted = False
    finish_reason = None
    try:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get('error'):
                raise LLMError('Provider returned a streaming error.')
            choices = chunk.get('choices') or [{}]
            choice = choices[0] if isinstance(choices[0], dict) else {}
            finish_reason = choice.get('finish_reason') or finish_reason
            content = choice.get('delta', {}).get('content')
            if isinstance(content, str):
                visible = parser.feed(content)
                if visible:
                    emitted = emitted or bool(visible.strip())
                    yield visible
        tail = parser.feed('', final=True)
        if tail:
            emitted = emitted or bool(tail.strip())
            yield tail
        if finish_reason == 'length':
            raise LLMError('LLM output token budget exhausted before completion.')
        if not emitted:
            raise LLMError('LLM returned no final answer; reasoning text is not an answer.')
    finally:
        close = getattr(resp, 'close', None)
        if close:
            close()


def generate(system_prompt, user_prompt, temperature=0.1, max_tokens=2048, *, thinking=False, reasoning_mode=None):
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
        **({"reasoning_mode": reasoning_mode} if reasoning_mode is not None else {}),
    )

    from src.generation.reasoning import final_text
    content = final_text(original_content)
    if not content:
        raise LLMError("LLM returned no final answer; reasoning text is not an answer.")
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
        resp = post_json(
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
        if response["choices"][0].get("finish_reason") == "length":
            raise LLMError("Vision LLM output token budget exhausted before completion.")
        original_content = response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError("Vision LLM returned an unexpected response structure.") from e

    if isinstance(original_content, list):
        # Some servers return content as a list of parts
        parts = []
        for part in original_content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text") or "")
            elif isinstance(part, str):
                parts.append(part)
        original_content = "".join(parts)

    from src.generation.reasoning import final_text
    content = final_text(original_content)
    if not content:
        raise LLMError("Vision LLM returned no final answer; reasoning text is not an answer.")
    return content


def parse_json_response(text: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences and thinking blocks."""
    if not text:
        raise ValueError("Empty response text")

    from src.generation.reasoning import final_text
    text = final_text(text)
    if not text:
        raise ValueError("No final content found after stripping reasoning")

    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    if not text:
        raise ValueError("No content remaining after stripping formatting")

    return json.loads(text)
