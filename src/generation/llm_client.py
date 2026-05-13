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

    logger.info(f"Curl exit code: {result.returncode}, stdout length: {len(result.stdout)}, stderr: {result.stderr[:100] if result.stderr else 'none'}")

    if result.returncode != 0:
        error_msg = f"LLM request failed (curl exit {result.returncode}): {result.stderr[:500]}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Save raw response to file for inspection
    import time
    raw_response_file = f"/tmp/llm_raw_response_{int(time.time())}.txt"
    try:
        with open(raw_response_file, 'w') as f:
            f.write(result.stdout)
        logger.info(f"Raw response saved to: {raw_response_file}")
    except Exception as e:
        logger.error(f"Failed to save raw response: {e}")

    try:
        response = json.loads(result.stdout)
        logger.info(f"LLM response keys: {list(response.keys())}")
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON from LLM: {e}\nResponse: {result.stdout[:500]}"
        logger.error(error_msg)
        logger.error(f"Raw response file: {raw_response_file}")
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

    # Fallback 1: some models put output in 'reasoning' or 'reasoning_content' field
    if not content:
        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
        if reasoning:
            logger.info(f"Using reasoning field as content fallback, length: {len(reasoning)}")
            content = reasoning.strip()

    # Fallback 2: try to extract text from thinking tags if content is empty
    if not content and 'content' in message:
        raw = message.get('content', '')
        logger.debug(f"Attempting to extract from thinking block in content, raw length: {len(raw)}")
        # Extract text from <think>...</think> blocks if that's all there is
        think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            if thinking_content:
                logger.info(f"Extracted content from thinking block, length: {len(thinking_content)}")
                content = thinking_content
            # Also check if there's content after the thinking block
            after_think = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            if after_think and len(after_think) > len(thinking_content):
                content = after_think
                logger.info(f"Using content after thinking block, length: {len(after_think)}")

    # Fallback 3: try other common field names
    if not content:
        for field_name in ['text', 'output', 'result', 'message', 'data']:
            if field_name in message:
                field_value = message.get(field_name, '')
                if isinstance(field_value, str) and field_value.strip():
                    content = field_value.strip()
                    logger.info(f"Using '{field_name}' field as content fallback, length: {len(content)}")
                    break

    # Fallback 4: Last resort - if we have a content field but it's only thinking blocks,
    # return the thinking content (it's better than an error)
    if not content and 'content' in message:
        raw = message.get('content', '')
        if raw:
            think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
            if think_match:
                thinking_text = think_match.group(1).strip()
                if thinking_text:
                    logger.warning(f"FALLBACK: Returning thinking block content ({len(thinking_text)} chars) - no other content found")
                    content = thinking_text

    if not content:
        # Log comprehensive debugging information
        logger.error(f"CRITICAL: LLM returned empty content after all fallbacks")
        logger.error(f"Model: {model}")
        logger.error(f"Message keys: {list(message.keys())}")
        logger.error(f"Message: {json.dumps(message, indent=2)}")

        # Also write to a file for easy inspection
        import time
        debug_file = f"/tmp/llm_debug_{int(time.time())}.json"
        try:
            with open(debug_file, 'w') as f:
                json.dump({
                    'model': model,
                    'message_keys': list(message.keys()),
                    'message': message,
                    'full_response': result.stdout,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode,
                }, f, indent=2)
            logger.error(f"Full debug info written to: {debug_file}")
            print(f"\n\n=== DEBUG FILE CREATED: {debug_file} ===\n\n", flush=True)
        except Exception as e:
            logger.error(f"Failed to write debug file: {e}")

        error_msg = f"LLM returned empty content.\nMessage keys: {list(message.keys())}\nDebug file: {debug_file}"
        raise RuntimeError(error_msg)

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

    proc = subprocess.Popen(
        ['curl', '-4', '-s', '-N', '-X', 'POST',
         f'{settings.vllm_base_url}/chat/completions',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps(payload)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    buffer = ""
    for line in proc.stdout:
        line = line.strip()
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]  # strip "data: " prefix
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                # Strip thinking blocks from streamed content
                buffer += content
                # Only yield content outside <think> blocks
                while "<think>" in buffer and "</think>" in buffer:
                    start = buffer.index("<think>")
                    end = buffer.index("</think>") + len("</think>")
                    buffer = buffer[:start] + buffer[end:]
                # If we're inside an unclosed <think> block, don't yield yet
                if "<think>" in buffer and "</think>" not in buffer:
                    continue
                if buffer:
                    yield buffer
                    buffer = ""
        except json.JSONDecodeError:
            continue

    proc.wait()
    if buffer:
        # Flush any remaining non-thinking content
        buffer = re.sub(r"<think>.*?</think>", "", buffer, flags=re.DOTALL).strip()
        if buffer:
            yield buffer


def generate(system_prompt, user_prompt, temperature=0.1, max_tokens=2048):
    """Generate text using the LLM with IPv4 forcing to avoid VPN timeouts."""
    original_content = _call_llm_with_curl(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=settings.vllm_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    logger.debug(f"Original content length: {len(original_content)}, preview: {original_content[:100]}")

    # Strategy 1: Strip <think>...</think> blocks but preserve content outside them
    content = re.sub(r"<think>.*?</think>", "", original_content, flags=re.DOTALL).strip()
    logger.debug(f"Content after stripping think blocks: length={len(content)}, preview: {content[:100] if content else 'EMPTY'}")

    # Strategy 2: If stripping thinking blocks left nothing, the entire response was in thinking
    # In that case, extract from the thinking block (it's better than an error)
    if not content:
        logger.warning("All content was in thinking blocks, attempting to extract from thinking")
        think_match = re.search(r'<think>(.*?)</think>', original_content, re.DOTALL)
        if think_match:
            thinking_text = think_match.group(1).strip()
            if thinking_text:
                logger.info(f"Extracted from thinking block: {len(thinking_text)} chars")
                content = thinking_text
            else:
                logger.warning("Thinking block was empty")

    # Strategy 3: If still no content, return the original (it's all we have)
    if not content:
        logger.warning(f"Could not extract content, using original response as-is (length: {len(original_content)})")
        content = original_content.strip()

    if not content:
        error_msg = f"LLM returned completely empty response"
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
