import json
import logging
import re

import requests

from src.config import settings

logger = logging.getLogger(__name__)


def _call_llm(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    """Call LLM via requests to an OpenAI-compatible endpoint."""
    logger.info(f"LLM call: model={model}, temperature={temperature}, max_tokens={max_tokens}")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(
            f'{settings.vllm_base_url}/chat/completions',
            json=payload,
            timeout=settings.vllm_request_timeout,
        )
        resp.raise_for_status()
        response = resp.json()
    except requests.ConnectionError as e:
        raise RuntimeError(f"LLM connection failed: {e}")
    except requests.Timeout:
        raise RuntimeError(f"LLM request timed out after {settings.vllm_request_timeout}s")
    except requests.HTTPError as e:
        raise RuntimeError(f"LLM HTTP error: {e}")
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


def generate_stream(system_prompt, user_prompt, temperature=0.1, max_tokens=2048):
    """Stream tokens from the LLM. Yields content strings as they arrive."""
    payload = {
        "model": settings.vllm_model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    resp = requests.post(
        f'{settings.vllm_base_url}/chat/completions',
        json=payload,
        stream=True,
        timeout=settings.vllm_request_timeout,
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
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
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


def generate(system_prompt, user_prompt, temperature=0.1, max_tokens=2048):
    """Generate text using the LLM."""
    original_content = _call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=settings.vllm_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
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
