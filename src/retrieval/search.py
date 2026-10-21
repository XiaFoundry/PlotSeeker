"""Unified keyword, vector and hybrid subject retrieval service."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from config import MODEL_CACHE_DIR, SQLITE_PATH, VECTOR_STORE_DIR
from retrieval.fusion import (
    DEFAULT_RRF_K,
    hybrid_candidates,
    keyword_candidates,
    vector_candidates,
)
from retrieval.keyword_index import KEYWORD_DB_PATH, KeywordSearcher
from retrieval.preprocess import normalize_text
from retrieval.relations import SubjectResolver
from retrieval.vector_search import VectorSearcher

MODES = ("keyword", "vector", "hybrid")
MAX_QUERY_CHARS = 2000
KEYWORD_COVERAGE_THRESHOLD = 0.75


class RetrievalService:
    def __init__(
        self,
        *,
        keyword_db: Path = KEYWORD_DB_PATH,
        store_dir: Path = VECTOR_STORE_DIR,
        checkpoint_path: Path | None = None,
        model_cache: Path = MODEL_CACHE_DIR / "bge-m3",
        sqlite_path: Path = SQLITE_PATH,
        vector_embedder=None,
        device: str = "auto",
        keyword_tokenizer: str | None = None,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.keyword_db = Path(keyword_db)
        self.store_dir = Path(store_dir)
        self.checkpoint_path = checkpoint_path
        self.model_cache = Path(model_cache)
        self.sqlite_path = Path(sqlite_path)
        self.vector_embedder = vector_embedder
        self.device = device
        self.keyword_tokenizer = keyword_tokenizer
        self.rrf_k = rrf_k
        self._keyword: KeywordSearcher | None = None
        self._vector: VectorSearcher | None = None
        self._resolver: SubjectResolver | None = None
        self._lock = threading.RLock()

    def _keyword_searcher(self) -> KeywordSearcher:
        if self._keyword is None:
            self._keyword = KeywordSearcher(
                self.keyword_db, tokenizer=self.keyword_tokenizer
            )
        return self._keyword

    def _vector_searcher(self) -> VectorSearcher:
        if self._vector is None:
            self._vector = VectorSearcher(
                store_dir=self.store_dir,
                checkpoint_path=self.checkpoint_path,
                model_cache=self.model_cache,
                embedder=self.vector_embedder,
                device=self.device,
            )
        return self._vector

    def _subject_resolver(self) -> SubjectResolver:
        if self._resolver is None:
            self._resolver = SubjectResolver(self.sqlite_path)
        return self._resolver

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        top_k: int = 10,
        candidate_k: int = 50,
    ) -> dict[str, Any]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of: {', '.join(MODES)}")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query exceeds {MAX_QUERY_CHARS} characters")
        if not normalize_text(query):
            raise ValueError("query is empty")

        with self._lock:
            started = time.perf_counter()
            keyword_hits = []
            vector_hits = []
            keyword_ms = 0.0
            vector_ms = 0.0
            resolver = self._subject_resolver()

            if mode in {"keyword", "hybrid"}:
                keyword_started = time.perf_counter()
                keyword_hits = self._keyword_searcher().search(query, top_k=candidate_k)
                keyword_ms = (time.perf_counter() - keyword_started) * 1000.0

            if mode in {"vector", "hybrid"}:
                vector_started = time.perf_counter()
                vector_hits, vector_ms = self._vector_searcher().search_timed(
                    query, top_k_per_collection=candidate_k
                )

            if mode == "keyword":
                candidates = keyword_candidates(
                    keyword_hits, resolver, rrf_k=self.rrf_k
                )
            elif mode == "vector":
                candidates = vector_candidates(
                    vector_hits, resolver, rrf_k=self.rrf_k
                )
            else:
                candidates = hybrid_candidates(
                    keyword_hits, vector_hits, resolver, rrf_k=self.rrf_k
                )

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            keyword_coverage = max(
                (hit.coverage or 0.0 for hit in keyword_hits), default=0.0
            )
            keyword_good = bool(keyword_hits) and (
                keyword_coverage >= KEYWORD_COVERAGE_THRESHOLD
            )
            if not candidates:
                confidence = "none"
            elif keyword_good and vector_hits:
                confidence = "high"
            elif keyword_good:
                confidence = "medium"
            else:
                confidence = "low"
            vector_top_similarity = (
                round(max((hit.score for hit in vector_hits), default=0.0), 4)
                if vector_hits
                else None
            )
            warnings = []
            if keyword_hits and not keyword_good:
                warnings.append("weak_keyword_coverage")
            if not keyword_hits and vector_hits:
                warnings.append("vector_only_no_keyword_match")
            return {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "rrf_k": self.rrf_k,
                "confidence": confidence,
                "warnings": warnings,
                "keyword_coverage": round(keyword_coverage, 4),
                "vector_top_similarity": vector_top_similarity,
                "counts": {
                    "keyword_hits": len(keyword_hits),
                    "vector_hits": len(vector_hits),
                    "subject_candidates": len(candidates),
                },
                "timings_ms": {
                    "keyword": round(keyword_ms, 2),
                    "vector": round(vector_ms, 2),
                    "total": round(elapsed_ms, 2),
                },
                "candidates": [item.to_dict() for item in candidates[:top_k]],
            }

    def close(self) -> None:
        with self._lock:
            if self._keyword is not None:
                self._keyword.close()
                self._keyword = None
            if self._vector is not None:
                self._vector.close()
                self._vector = None
            if self._resolver is not None:
                self._resolver.close()
                self._resolver = None

    def __enter__(self) -> "RetrievalService":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
