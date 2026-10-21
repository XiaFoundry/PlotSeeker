from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from api.app import app, get_service


class FakeService:
    def __init__(self) -> None:
        self.closed = False

    def search(self, query: str, *, mode: str, top_k: int, candidate_k: int):
        if query == "slow":
            time.sleep(0.2)
        if query == "error":
            raise RuntimeError("boom")
        if query == "missing":
            raise FileNotFoundError("index missing")
        return {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "rrf_k": 60,
            "confidence": "high" if mode == "hybrid" else "medium",
            "warnings": [],
            "keyword_coverage": 1.0,
            "counts": {
                "keyword_hits": 3,
                "vector_hits": 3 if mode != "keyword" else 0,
                "subject_candidates": 1,
            },
            "timings_ms": {"keyword": 1.0, "vector": 2.0, "total": 3.0},
            "candidates": [
                {
                    "subject_id": 1,
                    "name": "時間循環学園",
                    "name_cn": "时间循环学园",
                    "score": 0.03,
                    "keyword_rank": 1,
                    "vector_rank": 1 if mode != "keyword" else None,
                    "evidence": [
                        {
                            "doc_id": "episode:1",
                            "entity_type": "episode",
                            "entity_id": 1,
                            "subject_id": 1,
                            "source": "keyword",
                            "rank": 1,
                            "score": 1.0,
                            "snippet": "妹妹用脚抵住门。",
                            "relation_path": "episode:1->subject:1",
                            "contribution": 0.016,
                        }
                    ],
                }
            ],
        }

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def client():
    fake = FakeService()
    app.dependency_overrides[get_service] = lambda: fake
    app.state.search_timeout_seconds = 5.0
    with TestClient(app) as test_client:
        test_client.fake_service = fake
        yield test_client
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "p8-v1"
    assert set(body) >= {
        "status",
        "keyword_index",
        "vector_index",
        "model_cache",
        "sqlite",
    }


def test_search_returns_candidates_and_evidence(client):
    response = client.get(
        "/search", params={"q": "妹妹用脚抵住门", "mode": "hybrid", "top_k": 5}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "妹妹用脚抵住门"
    assert body["mode"] == "hybrid"
    assert body["confidence"] == "high"
    assert body["candidates"][0]["subject_id"] == 1
    evidence = body["candidates"][0]["evidence"][0]
    assert evidence["relation_path"] == "episode:1->subject:1"


@pytest.mark.parametrize(
    "params",
    [
        {"q": "   "},
        {"q": "x", "top_k": 0},
        {"q": "x", "candidate_k": 0},
        {"q": "x", "mode": "unknown"},
        {"q": "x" * 2001},
    ],
)
def test_invalid_parameters_return_422(client, params):
    response = client.get("/search", params=params)
    assert response.status_code == 422


def test_timeout_returns_504(client):
    app.state.search_timeout_seconds = 0.05
    response = client.get("/search", params={"q": "slow", "mode": "keyword"})
    assert response.status_code == 504


def test_missing_index_returns_503(client):
    response = client.get("/search", params={"q": "missing", "mode": "keyword"})
    assert response.status_code == 503


def test_unexpected_error_returns_500(client):
    response = client.get("/search", params={"q": "error", "mode": "keyword"})
    assert response.status_code == 500
