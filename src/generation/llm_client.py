import json
import logging
import re
import subprocess

from src.config import settings

logger = logging.getLogger(__name__)


def _call_llm_with_curl(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    """Call LLM using curl with IPv4 forcing to avoid VPN timeout issues."""

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
        timeout=60
    )

    if result.returncode != 0:
        error_msg = f"LLM request failed (curl exit {result.returncode}): {result.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        response = json.loads(result.stdout)
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

    return response['choices'][0]['message']['content']


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
    # Strip <think>...</think> blocks from thinking models
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def parse_json_response(text: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences and thinking blocks."""
    # Strip thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    return json.loads(text)
