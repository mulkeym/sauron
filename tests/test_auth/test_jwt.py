# tests/test_auth/test_jwt.py
import time
import pytest
from src.auth.jwt import create_token, decode_token
from src.auth.models import UserContext


def test_create_and_decode_token():
    token = create_token(username="mike", groups=["finance", "executives"])
    user = decode_token(token)
    assert user.username == "mike"
    assert "finance" in user.groups
    assert "executives" in user.groups


def test_decode_token_expired():
    token = create_token(username="mike", groups=[], expiration_minutes=-1)
    with pytest.raises(ValueError, match="expired"):
        decode_token(token)


def test_decode_token_invalid():
    with pytest.raises(ValueError, match="Invalid token"):
        decode_token("not-a-valid-token")


def test_create_token_contains_groups():
    token = create_token(username="bob", groups=["it_support"])
    user = decode_token(token)
    assert user.groups == ["it_support"]
