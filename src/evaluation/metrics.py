"""Retrieval metric helpers used by the P6 evaluation runner."""

from __future__ import annotations

from typing import Sequence


def first_relevant_rank(
    retrieved: Sequence[int], acceptable: Sequence[int], limit: int
) -> int | None:
    acceptable_set = {int(value) for value in acceptable}
    for index, subject_id in enumerate(retrieved[:limit], 1):
        if int(subject_id) in acceptable_set:
            return index
    return None


def recall_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def top1_accuracy(rank: int | None) -> float:
    return 1.0 if rank == 1 else 0.0


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * p / 100.0))
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]
