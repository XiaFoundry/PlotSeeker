from __future__ import annotations

import pytest

from evaluation.metrics import (
    first_relevant_rank,
    percentile,
    recall_at_k,
    reciprocal_rank,
    top1_accuracy,
)
from evaluation.runner import _aggregate, _failure_category


def test_rank_and_metrics():
    retrieved = [10, 20, 30, 40]
    assert first_relevant_rank(retrieved, [30], 10) == 3
    assert first_relevant_rank(retrieved, [99], 10) is None
    assert recall_at_k(3, 5) == 1.0
    assert recall_at_k(6, 5) == 0.0
    assert reciprocal_rank(4, 10) == 0.25
    assert reciprocal_rank(None, 10) == 0.0
    assert top1_accuracy(1) == 1.0
    assert top1_accuracy(2) == 0.0


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0


def test_aggregate_separates_answerable_and_negative():
    base = {
        "clue_type": "name",
        "latency_ms": 10.0,
    }
    records = [
        {
            **base,
            "answerable": True,
            "recall_at_5": 1.0,
            "mrr_at_10": 1.0,
            "top1": 1.0,
            "top_ids": [1],
            "confidence": "high",
        },
        {
            **base,
            "answerable": False,
            "recall_at_5": 0.0,
            "mrr_at_10": 0.0,
            "top1": 0.0,
            "top_ids": [2],
            "confidence": "low",
        },
    ]
    summary = _aggregate(records)
    assert summary["answerable_queries"] == 1
    assert summary["negative_queries"] == 1
    assert summary["recall_at_5"] == 1.0
    assert summary["negative_false_positive_any"] == 1
    assert summary["negative_false_positive_confident"] == 0


@pytest.mark.parametrize(
    "record,expected",
    [
        (
            {"failure_reason": "keyword_not_in_top10", "clue_type": "name"},
            "名称/别名/简繁",
        ),
        (
            {"failure_reason": "vector_not_in_top10", "clue_type": "subject_summary"},
            "向量召回",
        ),
        (
            {"failure_reason": "no_candidates", "clue_type": "fragment"},
            "数据缺失或文本质量",
        ),
        (
            {"failure_reason": "confident_false_positive", "clue_type": "unanswerable"},
            "问题歧义或负样例",
        ),
    ],
)
def test_failure_category(record, expected):
    assert _failure_category(record) == expected
