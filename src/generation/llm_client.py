import json
import re
from functools import lru_cache

from openai import OpenAI

from src.config import settings


@lru_cache(maxsize=1)
def _get_client():
    return OpenAI(base_url=settings.vllm_base_url, api_key="not-needed")


def generate(system_prompt, user_prompt, temperature=0.1, max_tokens=2048):
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.vllm_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
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
