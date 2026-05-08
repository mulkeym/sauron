# src/auth/api_key.py
from src.config import settings


def validate_api_key(key: str) -> bool:
    if not key:
        return False
    return key in settings.api_key_list
