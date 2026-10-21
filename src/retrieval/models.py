"""Small data objects shared by keyword, vector and hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentHit:
    doc_id: str
    entity_type: str
    entity_id: int
    subject_id: int
    text: str
    metadata: dict[str, Any]
    score: float
    rank: int
    source: str
    snippet: str = ""
    relation_path: str | None = None
    collection_rank: int | None = None
    coverage: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "subject_id": self.subject_id,
            "score": self.score,
            "rank": self.rank,
            "source": self.source,
            "snippet": self.snippet,
            "relation_path": self.relation_path,
            "collection_rank": self.collection_rank,
            "coverage": None if self.coverage is None else round(self.coverage, 4),
        }


@dataclass
class Evidence:
    doc_id: str
    entity_type: str
    entity_id: int
    subject_id: int
    source: str
    rank: int
    score: float
    snippet: str
    relation_path: str
    contribution: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "subject_id": self.subject_id,
            "source": self.source,
            "rank": self.rank,
            "score": round(self.score, 6),
            "snippet": self.snippet,
            "relation_path": self.relation_path,
            "contribution": round(self.contribution, 6),
        }


@dataclass
class SubjectCandidate:
    subject_id: int
    name: str
    name_cn: str
    score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None
    evidence: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "name": self.name,
            "name_cn": self.name_cn,
            "score": round(self.score, 6),
            "keyword_rank": self.keyword_rank,
            "vector_rank": self.vector_rank,
            "evidence": [item.to_dict() for item in self.evidence],
        }
