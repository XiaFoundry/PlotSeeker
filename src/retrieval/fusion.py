"""Reciprocal Rank Fusion and subject-level evidence assembly."""

from __future__ import annotations

from typing import Iterable, Sequence

from retrieval.models import DocumentHit, Evidence, SubjectCandidate
from retrieval.relations import SubjectResolver

DEFAULT_RRF_K = 60
MAX_EVIDENCE_PER_CANDIDATE = 6
MAX_EVIDENCE_PER_SOURCE = 3


def rrf_score(rank: int, k: int = DEFAULT_RRF_K) -> float:
    if rank <= 0:
        raise ValueError("rank must be positive")
    return 1.0 / (k + rank)


def fuse_candidates(
    keyword_hits: Sequence[DocumentHit],
    vector_hits: Sequence[DocumentHit],
    resolver: SubjectResolver,
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[SubjectCandidate]:
    best: dict[int, dict[str, float]] = {}
    best_rank: dict[int, dict[str, int]] = {}
    evidence_by_subject: dict[int, list[Evidence]] = {}

    for source_name, hits in (("keyword", keyword_hits), ("vector", vector_hits)):
        for hit in hits:
            for subject_id, relation_path, weight in resolver.resolve_hit(hit):
                if resolver.subject(subject_id) is None:
                    continue
                contribution = rrf_score(hit.rank, rrf_k) * weight
                source_scores = best.setdefault(subject_id, {})
                source_ranks = best_rank.setdefault(subject_id, {})
                if contribution > source_scores.get(source_name, 0.0):
                    source_scores[source_name] = contribution
                current_rank = source_ranks.get(source_name)
                if current_rank is None or hit.rank < current_rank:
                    source_ranks[source_name] = hit.rank
                evidence_by_subject.setdefault(subject_id, []).append(
                    Evidence(
                        doc_id=hit.doc_id,
                        entity_type=hit.entity_type,
                        entity_id=hit.entity_id,
                        subject_id=subject_id,
                        source=source_name,
                        rank=hit.rank,
                        score=hit.score,
                        snippet=hit.snippet,
                        relation_path=relation_path,
                        contribution=contribution,
                    )
                )

    candidates: list[SubjectCandidate] = []
    for subject_id, source_scores in best.items():
        subject = resolver.subject(subject_id)
        if subject is None:
            continue
        evidence = evidence_by_subject.get(subject_id, [])
        evidence.sort(key=lambda item: (-item.contribution, item.rank, item.doc_id))
        per_source: dict[str, int] = {}
        selected: list[Evidence] = []
        for item in evidence:
            count = per_source.get(item.source, 0)
            if count >= MAX_EVIDENCE_PER_SOURCE:
                continue
            selected.append(item)
            per_source[item.source] = count + 1
            if len(selected) >= MAX_EVIDENCE_PER_CANDIDATE:
                break
        ranks = best_rank.get(subject_id, {})
        candidates.append(
            SubjectCandidate(
                subject_id=subject_id,
                name=subject["name"],
                name_cn=subject["name_cn"],
                score=sum(source_scores.values()),
                keyword_rank=ranks.get("keyword"),
                vector_rank=ranks.get("vector"),
                evidence=selected,
            )
        )
    candidates.sort(key=lambda item: (-item.score, item.subject_id))
    return candidates


def keyword_candidates(
    hits: Sequence[DocumentHit],
    resolver: SubjectResolver,
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[SubjectCandidate]:
    return fuse_candidates(hits, (), resolver, rrf_k=rrf_k)


def vector_candidates(
    hits: Sequence[DocumentHit],
    resolver: SubjectResolver,
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[SubjectCandidate]:
    return fuse_candidates((), hits, resolver, rrf_k=rrf_k)


def hybrid_candidates(
    keyword_hits: Sequence[DocumentHit],
    vector_hits: Sequence[DocumentHit],
    resolver: SubjectResolver,
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[SubjectCandidate]:
    return fuse_candidates(keyword_hits, vector_hits, resolver, rrf_k=rrf_k)
