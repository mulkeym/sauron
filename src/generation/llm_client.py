import json
import logging
import re
import subprocess

from src.config import settings

logger = logging.getLogger(__name__)


def _call_llm_with_curl(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    """Call LLM using curl with IPv4 forcing to avoid VPN timeout issues."""

    logger.info(f"LLM call: model={model}, temperature={temperature}, max_tokens={max_tokens}")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    result = subprocess.run(
        ['curl', '-4', '-s', '-X', 'POST',
         f'{settings.vllm_base_url}/chat/completions',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=settings.vllm_request_timeout
    )

    if result.returncode != 0:
        error_msg = f"LLM request failed (curl exit {result.returncode}): {result.stderr[:500]}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        response = json.loads(result.stdout)
        logger.debug(f"LLM response keys: {list(response.keys())}")
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON from LLM: {e}\nResponse: {result.stdout[:500]}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if 'error' in response:
        error_msg = f"LLM error: {response['error']}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if 'choices' not in response or not response['choices']:
        error_msg = f"No choices in LLM response: {result.stdout[:200]}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    message = response['choices'][0]['message']
    logger.debug(f"Message keys: {list(message.keys())}")
    logger.debug(f"Message content field length: {len(message.get('content', ''))}")

    content = message.get('content', '').strip() if message.get('content') else ''
    logger.debug(f"Content after strip: '{content[:100] if content else 'EMPTY'}'")

    # Fallback 1: some models put output in 'reasoning' field
    if not content and 'reasoning' in message:
        reasoning = message.get('reasoning', '')
        logger.debug(f"Has reasoning field, length: {len(reasoning)}")
        content = reasoning.strip() if reasoning else ''
        if content:
            logger.info("Using reasoning field as content fallback")

    # Fallback 2: try to extract text from thinking tags if content is empty
    if not content and 'content' in message:
        raw = message.get('content', '')
        logger.debug(f"Attempting to extract from thinking block in content, raw length: {len(raw)}")
        # Extract text from <think>...</think> blocks if that's all there is
        think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
        if think_match:
            content = think_match.group(1).strip()
            if content:
                logger.info(f"Extracted content from thinking block, length: {len(content)}")

    # Fallback 3: try other common field names
    if not content:
        for field_name in ['text', 'output', 'result', 'message']:
            if field_name in message:
                field_value = message.get(field_name, '')
                if isinstance(field_value, str) and field_value.strip():
                    content = field_value.strip()
                    logger.info(f"Using '{field_name}' field as content fallback, length: {len(content)}")
                    break

    if not content:
        # Log comprehensive debugging information
        logger.error(f"CRITICAL: LLM returned empty content")
        logger.error(f"Model: {model}")
        logger.error(f"Message keys: {list(message.keys())}")
        logger.error(f"Message content: {json.dumps(message, indent=2)}")
        logger.error(f"Full response: {result.stdout[:2000]}")

        error_msg = f"LLM returned empty content.\nMessage keys: {list(message.keys())}\nMessage: {json.dumps(message, indent=2)[:1000]}\nFull response: {result.stdout[:1000]}"
        raise RuntimeError(error_msg)

    return content


def generate(system_prompt, user_prompt, temperature=0.1, max_tokens=2048):
    """Generate text using the LLM with IPv4 forcing to avoid VPN timeouts."""
    content = _call_llm_with_curl(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=settings.vllm_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Strip <think>...</think> blocks from thinking models (in case there's content outside blocks)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    if not content:
        # This shouldn't happen if _call_llm_with_curl is working correctly,
        # but if it does, give a better error message
        error_msg = f"LLM returned empty content after processing"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return content


def parse_json_response(text: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences and thinking blocks."""
    if not text:
        raise ValueError("Empty response text")

    # Strip thinking blocks - but if that's all we have, extract from within them
    original_text = text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if not text:
        # If stripping thinking blocks left nothing, try extracting from the block
        think_match = re.search(r'<think>(.*?)</think>', original_text, re.DOTALL)
        if think_match:
            text = think_match.group(1).strip()
            logger.info("Extracted JSON from thinking block")
        else:
            raise ValueError("No valid content found after stripping thinking blocks")

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    if not text:
        raise ValueError("No content remaining after stripping formatting")

    return json.loads(text)
