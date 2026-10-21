"""Tool layer for the P9 agent.

Every tool reads an existing P2/P4/P5 artifact and returns traceable document
or subject IDs.  No tool fabricates an entity: missing subject rows are skipped
rather than inventing names.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from config import MODEL_CACHE_DIR, SQLITE_PATH, VECTOR_STORE_DIR
from retrieval.keyword_index import KEYWORD_DB_PATH, KeywordSearcher
from retrieval.relations import SubjectResolver
from retrieval.search import KEYWORD_COVERAGE_THRESHOLD, RetrievalService
from retrieval.vector_search import VectorSearcher

RRF_K = 60

TOOL_SPECS: dict[str, dict[str, Any]] = {
    "search_subjects": {
        "description": "Search works by fuzzy plot, title, alias or world setting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
        "output": "subject candidates with evidence",
    },
    "search_characters": {
        "description": "Search character documents and map them to their works.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
        "output": "character document hits with subject mapping",
    },
    "search_episodes": {
        "description": "Search episode descriptions and map them to their works.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
        "output": "episode document hits with subject mapping",
    },
    "get_related_subjects": {
        "description": "Return related works from the P2 subject_relations table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject_id": {"type": "integer"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["subject_id"],
        },
        "output": "related subjects with raw relation_type",
    },
    "ask_user_for_clarification": {
        "description": "Ask for more clues when retrieval is insufficient or ambiguous.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["question"],
        },
        "output": "structured clarification request",
    },
}


class AgentTools:
    def __init__(
        self,
        *,
        keyword_db: Path = KEYWORD_DB_PATH,
        store_dir: Path = VECTOR_STORE_DIR,
        checkpoint_path: Path | None = None,
        model_cache: Path = MODEL_CACHE_DIR / "bge-m3",
        sqlite_path: Path = SQLITE_PATH,
        embedder=None,
        device: str = "auto",
        candidate_k: int = 50,
    ) -> None:
        self.candidate_k = candidate_k
        self.retrieval = RetrievalService(
            keyword_db=keyword_db,
            store_dir=store_dir,
            checkpoint_path=checkpoint_path,
            model_cache=model_cache,
            sqlite_path=sqlite_path,
            vector_embedder=embedder,
            device=device,
        )
        self.keyword = KeywordSearcher(keyword_db)
        self.vector = VectorSearcher(
            store_dir=store_dir,
            checkpoint_path=checkpoint_path,
            model_cache=model_cache,
            embedder=embedder,
            device=device,
        )
        self.sqlite_path = Path(sqlite_path)
        self.resolver = SubjectResolver(self.sqlite_path)
        self._connection = sqlite3.connect(
            f"file:{self.sqlite_path}?mode=ro", uri=True, check_same_thread=False
        )

    def specs(self) -> dict[str, dict[str, Any]]:
        return TOOL_SPECS

    def search_subjects(
        self, query: str, top_k: int = 5, mode: str = "hybrid"
    ) -> dict[str, Any]:
        result = self.retrieval.search(
            query, mode=mode, top_k=top_k, candidate_k=self.candidate_k
        )
        return {
            "tool": "search_subjects",
            "query": query,
            "mode": mode,
            "confidence": result["confidence"],
            "warnings": result["warnings"],
            "count": len(result["candidates"]),
            "candidates": result["candidates"],
            "timings_ms": result["timings_ms"],
            "keyword_coverage": result["keyword_coverage"],
            "vector_top_similarity": result.get("vector_top_similarity"),
        }

    def _document_search(
        self, query: str, entity_type: str, top_k: int
    ) -> dict[str, Any]:
        keyword_hits = self.keyword.search(
            query, top_k=self.candidate_k, entity_types=[entity_type]
        )
        vector_hits = self.vector.search(
            query,
            top_k_per_collection=self.candidate_k,
            entity_types=[entity_type],
        )
        merged: dict[str, dict[str, Any]] = {}
        for hit in keyword_hits:
            entry = merged.setdefault(hit.doc_id, {"hit": hit})
            entry["keyword_rank"] = hit.rank
            entry["keyword_score"] = hit.score
            entry["coverage"] = hit.coverage
        for hit in vector_hits:
            entry = merged.setdefault(hit.doc_id, {"hit": hit})
            entry["vector_rank"] = hit.rank
            entry["vector_score"] = hit.score

        results: list[dict[str, Any]] = []
        for doc_id, entry in merged.items():
            hit = entry["hit"]
            keyword_rank = entry.get("keyword_rank")
            vector_rank = entry.get("vector_rank")
            score = 0.0
            if keyword_rank:
                score += 1.0 / (RRF_K + keyword_rank)
            if vector_rank:
                score += 1.0 / (RRF_K + vector_rank)
            subject = self.resolver.subject(hit.subject_id)
            if subject is None:
                continue
            results.append(
                {
                    "doc_id": doc_id,
                    "entity_type": hit.entity_type,
                    "entity_id": hit.entity_id,
                    "subject_id": hit.subject_id,
                    "subject_name": subject["name"],
                    "subject_name_cn": subject["name_cn"],
                    "name": (hit.metadata or {}).get("name", ""),
                    "keyword_rank": keyword_rank,
                    "vector_rank": vector_rank,
                    "keyword_score": entry.get("keyword_score"),
                    "vector_score": entry.get("vector_score"),
                    "coverage": entry.get("coverage"),
                    "score": round(score, 6),
                    "snippet": hit.snippet,
                    "relation_path": (
                        f"{hit.entity_type}:{hit.entity_id}"
                        f"->subject:{hit.subject_id}"
                    ),
                    "metadata_subject_ids": (hit.metadata or {}).get("subject_ids"),
                }
            )
        results.sort(key=lambda item: (-item["score"], item["doc_id"]))
        if results:
            max_coverage = max(
                (item.get("coverage") or 0.0 for item in results), default=0.0
            )
            keyword_good = max_coverage >= KEYWORD_COVERAGE_THRESHOLD
            if keyword_good and any(
                item["keyword_rank"] and item["vector_rank"] for item in results
            ):
                confidence = "high"
            elif keyword_good and any(item["keyword_rank"] for item in results):
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = "none"
        return {
            "tool": f"search_{entity_type}s" if entity_type != "episode" else "search_episodes",
            "query": query,
            "confidence": confidence,
            "count": len(results),
            "results": results[:top_k],
        }

    def search_characters(self, query: str, top_k: int = 5) -> dict[str, Any]:
        return self._document_search(query, "character", top_k)

    def search_episodes(self, query: str, top_k: int = 5) -> dict[str, Any]:
        return self._document_search(query, "episode", top_k)

    def get_related_subjects(
        self, subject_id: int, limit: int = 10
    ) -> dict[str, Any]:
        subject = self.resolver.subject(subject_id)
        if subject is None:
            return {
                "tool": "get_related_subjects",
                "subject_id": int(subject_id),
                "count": 0,
                "results": [],
                "error": "subject not found",
            }
        rows = self._connection.execute(
            """
            SELECT r.related_subject_id, s.name, s.name_cn, r.relation_type
            FROM subject_relations AS r
            JOIN subject AS s ON s.id = r.related_subject_id
            WHERE r.subject_id = ?
            ORDER BY r.ord, r.related_subject_id
            LIMIT ?
            """,
            (int(subject_id), max(1, int(limit))),
        ).fetchall()
        results = [
            {
                "subject_id": int(row[0]),
                "name": row[1] or "",
                "name_cn": row[2] or "",
                "relation_type": int(row[3]),
            }
            for row in rows
            if row[1] or row[2]
        ]
        return {
            "tool": "get_related_subjects",
            "subject_id": int(subject_id),
            "subject_name": subject["name"],
            "subject_name_cn": subject["name_cn"],
            "count": len(results),
            "results": results,
            "note": "relation_type is the raw P2 integer code.",
        }

    def ask_user_for_clarification(
        self,
        question: str,
        options: list[str] | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        return {
            "tool": "ask_user_for_clarification",
            "question": question,
            "options": options or [],
            "reason": reason,
        }

    def close(self) -> None:
        self.retrieval.close()
        self.keyword.close()
        self.vector.close()
        self.resolver.close()
        self._connection.close()

    def __enter__(self) -> "AgentTools":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
