from __future__ import annotations

import json

import pytest

from retrieval.keyword_index import KeywordSearcher, build_keyword_index
from retrieval.preprocess import (
    fts_query,
    normalize_text,
    tokenize_bigram,
    tokenize_jieba,
)


def test_preprocess_normalization_and_tokenizers():
    assert normalize_text("時間循環，校園！") == "时间循环 校园"
    jieba_tokens = tokenize_jieba("时间循环")
    bigram_tokens = tokenize_bigram("时间循环")
    assert "时间" in jieba_tokens
    assert {"时间", "间循", "循环"} <= set(bigram_tokens)
    assert fts_query(["时间", "循环"]) == '"时间" OR "循环"'
    assert fts_query([]) == ""


def test_keyword_index_search_filters_and_rebuild(retrieval_env):
    searcher = KeywordSearcher(retrieval_env["keyword_db"])
    try:
        metadata = searcher.metadata()
        assert metadata["format"] == "p5-keyword-v1"
        assert metadata["default_tokenizer"] in {"jieba", "bigram"}
        assert metadata["counts"] == {"subject": 3, "character": 3, "episode": 3}

        hits = searcher.search("时间循环", top_k=10)
        assert hits
        assert any(hit.doc_id == "subject:1" for hit in hits)
        assert any(hit.entity_type == "character" for hit in hits)
        assert all(hit.snippet for hit in hits)

        subject_hits = searcher.search("时间循环", top_k=10, entity_types=["subject"])
        assert subject_hits
        assert {hit.entity_type for hit in subject_hits} == {"subject"}

        assert searcher.search("zzzz-unfindable-xyz", top_k=10) == []
        document = searcher.document("subject:1")
        assert document is not None
        assert document["metadata"]["name"] == "时间循环学园"
    finally:
        searcher.close()

    with pytest.raises(FileExistsError):
        build_keyword_index(
            manifest_path=retrieval_env["manifest"],
            p4_checkpoint_path=retrieval_env["checkpoint"],
            output_path=retrieval_env["keyword_db"],
            report_path=retrieval_env["keyword_report"],
            verify_source=True,
            verbose=False,
        )

    rebuilt = build_keyword_index(
        manifest_path=retrieval_env["manifest"],
        p4_checkpoint_path=retrieval_env["checkpoint"],
        output_path=retrieval_env["keyword_db"],
        report_path=retrieval_env["keyword_report"],
        rebuild=True,
        verify_source=True,
        verbose=False,
    )
    assert rebuilt["document_count"] == 9
    assert rebuilt["verified"]["documents"] == 9
    assert set(rebuilt["verified"]["fts_counts"]) == {"jieba", "bigram"}


def test_keyword_report_contains_tokenizer_comparison(retrieval_env):
    report = json.loads(retrieval_env["keyword_report"].read_text("utf-8"))
    assert report["default_tokenizer"] in {"jieba", "bigram"}
    for tokenizer in ("jieba", "bigram"):
        entry = report["tokenizer_comparison"][tokenizer]
        assert entry["queries"] > 0
        assert 0.0 <= entry["recall_at_5"] <= 1.0
        assert 0.0 <= entry["mrr_at_10"] <= 1.0


def test_keyword_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        KeywordSearcher(tmp_path / "missing.db")
