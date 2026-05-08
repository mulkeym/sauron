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
    return response.choices[0].message.content
