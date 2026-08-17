from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from src.auth.jwt import create_token
from src.generation.rag_chain import RAGResponse
from src.main import create_app


def test_chat_completions_records_activity():
    token = create_token(username="mike", groups=["finance"])
    headers = {"Authorization": f"Bearer {token}", "X-API-Key": "test-key-1"}
    rec = AsyncMock()
    with patch("src.api.routes_openai_compat.agent_query", new_callable=AsyncMock, return_value=RAGResponse(answer="hi", citations=[], query_type="lookup", cached=False)), \
         patch("src.api.routes_openai_compat.get_vector_store", return_value=MagicMock()), \
         patch("src.api.routes_openai_compat.get_schema_registry", return_value=MagicMock()), \
         patch("src.audit.activity.record_query_activity", rec):
        resp = TestClient(create_app()).post(
            "/v1/chat/completions",
            json={"model": "sauron", "messages": [{"role": "user", "content": "hello"}]},
            headers=headers,
        )
    assert resp.status_code == 200
    rec.assert_awaited()
    kwargs = rec.await_args.kwargs
    assert kwargs["source"] == "openai"
    assert kwargs["tool"] == "chat.completions"
    assert kwargs["username"] == "mike"
    assert kwargs["user_groups"] == ["finance"]
    assert kwargs["query_text"] == "hello"
