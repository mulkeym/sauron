import pytest
from unittest.mock import patch, MagicMock


def test_generate_calls_openai_client():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    mock_client.chat.completions.create.return_value = mock_response
    with patch("src.generation.llm_client._get_client", return_value=mock_client):
        from src.generation.llm_client import generate
        result = generate(system_prompt="You are helpful.", user_prompt="Hello")
    assert result == "Test response"


def test_generate_passes_model_and_messages():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="answer"))]
    mock_client.chat.completions.create.return_value = mock_response
    with patch("src.generation.llm_client._get_client", return_value=mock_client):
        from src.generation.llm_client import generate
        generate(system_prompt="sys", user_prompt="usr")
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
