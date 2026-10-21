from __future__ import annotations

import pytest

from embeddings.fake import DeterministicFakeEmbedder
from retrieval.fusion import rrf_score
from retrieval.keyword_index import KeywordSearcher
from retrieval.search import RetrievalService

EPISODE_TEXT = (
    "所属作品：时间循环学园\n章节名称：门\n"
    "剧情：妹妹用脚抵住门，不让人进来。"
)


def _service(retrieval_env) -> RetrievalService:
    return RetrievalService(
        keyword_db=retrieval_env["keyword_db"],
        store_dir=retrieval_env["store_dir"],
        checkpoint_path=retrieval_env["checkpoint"],
        sqlite_path=retrieval_env["sqlite"],
        vector_embedder=DeterministicFakeEmbedder(retrieval_env["fake_dimension"]),
    )


def test_rrf_score():
    assert rrf_score(1, 60) == pytest.approx(1 / 61)
    assert rrf_score(10, 60) == pytest.approx(1 / 70)
    with pytest.raises(ValueError):
        rrf_score(0)


def test_keyword_episode_maps_to_subject_with_path(retrieval_env):
    service = _service(retrieval_env)
    try:
        result = service.search("妹妹用脚抵住门", mode="keyword", top_k=5)
    finally:
        service.close()
    assert result["candidates"]
    top = result["candidates"][0]
    assert top["subject_id"] == 1
    assert top["keyword_rank"] == 1
    assert top["vector_rank"] is None
    assert top["evidence"]
    assert any(
        item["doc_id"] == "episode:1"
        and item["relation_path"] == "episode:1->subject:1"
        for item in top["evidence"]
    )


def test_vector_exact_text_maps_to_subject(retrieval_env):
    service = _service(retrieval_env)
    try:
        result = service.search(EPISODE_TEXT, mode="vector", top_k=5)
    finally:
        service.close()
    assert result["candidates"]
    top = result["candidates"][0]
    assert top["subject_id"] == 1
    assert top["vector_rank"] is not None
    assert top["keyword_rank"] is None
    assert any(item["doc_id"] == "episode:1" for item in top["evidence"])


def test_hybrid_has_both_channels_and_evidence(retrieval_env):
    service = _service(retrieval_env)
    try:
        result = service.search(EPISODE_TEXT, mode="hybrid", top_k=5)
    finally:
        service.close()
    assert result["counts"]["keyword_hits"] > 0
    assert result["counts"]["vector_hits"] > 0
    top = result["candidates"][0]
    assert top["subject_id"] == 1
    assert top["keyword_rank"] is not None
    assert top["vector_rank"] is not None
    assert any(item["source"] == "keyword" for item in top["evidence"])
    assert any(item["source"] == "vector" for item in top["evidence"])


def test_character_secondary_relation_is_not_dominant(retrieval_env):
    service = _service(retrieval_env)
    try:
        result = service.search("感知时间循环", mode="keyword", top_k=5)
    finally:
        service.close()
    assert result["candidates"]
    top = result["candidates"][0]
    assert top["subject_id"] == 1
    character_evidence = [
        item for item in top["evidence"] if item["doc_id"] == "character:1"
    ]
    assert character_evidence
    assert character_evidence[0]["relation_path"] == "character:1->subject:1"

    secondary = [item for item in result["candidates"] if item["subject_id"] == 2]
    if secondary:
        assert secondary[0]["score"] < top["score"]


def test_keyword_negative_returns_no_candidate(retrieval_env):
    service = _service(retrieval_env)
    try:
        result = service.search("zzzz-unfindable-xyz", mode="keyword", top_k=5)
    finally:
        service.close()
    assert result["candidates"] == []


def test_vector_negative_results_are_real_documents(retrieval_env):
    service = _service(retrieval_env)
    try:
        result = service.search("zzzz-unfindable-xyz", mode="vector", top_k=5)
    finally:
        service.close()
    searcher = KeywordSearcher(retrieval_env["keyword_db"])
    try:
        for candidate in result["candidates"]:
            for evidence in candidate["evidence"]:
                assert searcher.document(evidence["doc_id"]) is not None
    finally:
        searcher.close()


def test_empty_and_long_queries_are_rejected(retrieval_env):
    service = _service(retrieval_env)
    try:
        with pytest.raises(ValueError):
            service.search("   ", mode="keyword")
        with pytest.raises(ValueError):
            service.search("x" * 2001, mode="keyword")
        with pytest.raises(ValueError):
            service.search("查询", mode="invalid")
    finally:
        service.close()
