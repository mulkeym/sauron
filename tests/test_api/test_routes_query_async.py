import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from src.main import create_app
from src.auth.jwt import create_token
from src.api.query_jobs import query_queue, QueryStatus, QueryJob


@pytest.fixture
def auth_headers():
    token = create_token(username="mike", groups=["finance"])
    return {"Authorization": f"Bearer {token}", "X-API-Key": "test-key-1"}


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _clear_queue():
    query_queue._jobs.clear()
    yield
    query_queue._jobs.clear()


def test_submit_returns_token_and_queued(client, auth_headers):
    with patch("src.api.routes_query.query_queue.start_worker", new_callable=AsyncMock):
        with patch("src.api.routes_query.get_vector_store", return_value=MagicMock()):
            with patch("src.api.routes_query.get_schema_registry", return_value=MagicMock()):
                with patch("src.api.routes_query.get_metadata_store", return_value=MagicMock()):
                    resp = client.post("/api/v1/query/async", json={"question": "slow q?"}, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"
    assert data["token"]
    job = query_queue.get_job(data["token"])
    assert job.username == "mike"
    assert job.question == "slow q?"


def test_submit_requires_auth(client):
    resp = client.post("/api/v1/query/async", json={"question": "x"})
    assert resp.status_code in (401, 403)


def test_poll_returns_completed_answer(client, auth_headers):
    query_queue._jobs["tok-1"] = QueryJob(
        token="tok-1", question="q", username="mike", groups=["finance"],
        status=QueryStatus.COMPLETE, step="complete", answer="done!",
        citations=[{"doc_id": "d1", "filename": "p.pdf", "doc_type": "pdf",
                    "chunk_index": 0, "page": 3, "snippet": "s", "relevance": 0.9}],
        completed_at=123.0,
    )
    resp = client.get("/api/v1/query/async/tok-1", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "complete"
    assert data["answer"] == "done!"
    assert data["citations"][0]["filename"] == "p.pdf"


def test_poll_unknown_token_404(client, auth_headers):
    resp = client.get("/api/v1/query/async/missing", headers=auth_headers)
    assert resp.status_code == 404


def test_poll_other_users_token_404(client, auth_headers):
    query_queue._jobs["tok-2"] = QueryJob(
        token="tok-2", question="q", username="someone-else", groups=["hr"],
        status=QueryStatus.PROCESSING, step="retrieving documents",
    )
    resp = client.get("/api/v1/query/async/tok-2", headers=auth_headers)
    assert resp.status_code == 404


def test_poll_requires_auth(client):
    resp = client.get("/api/v1/query/async/whatever")
    assert resp.status_code in (401, 403)
