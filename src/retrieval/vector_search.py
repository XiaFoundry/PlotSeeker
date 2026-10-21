"""Query-time adapter for the frozen P4 Chroma index."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from config import MODEL_CACHE_DIR, VECTOR_STORE_DIR
from embeddings.bge_m3 import BGEM3Embedder
from embeddings.chroma_store import create_client, query_collection
from retrieval.models import DocumentHit

DEFAULT_MODEL_CACHE = MODEL_CACHE_DIR / "bge-m3"
DEFAULT_COLLECTION_TOP_K = 50
ENTITY_TYPES = ("subject", "character", "episode")


class VectorIndexError(RuntimeError):
    pass


class VectorSearcher:
    def __init__(
        self,
        store_dir: Path = VECTOR_STORE_DIR,
        checkpoint_path: Path | None = None,
        model_cache: Path = DEFAULT_MODEL_CACHE,
        embedder=None,
        device: str = "auto",
        max_length: int | None = None,
        batch_size: int = 16,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.checkpoint_path = Path(checkpoint_path or self.store_dir / "p4_checkpoint.json")
        self.model_cache = Path(model_cache)
        self.device = device
        self.batch_size = batch_size
        self._embedder = embedder
        self._owns_embedder = embedder is None
        self._client = None
        self._checkpoint = self._load_checkpoint()
        self.collections = dict(self._checkpoint["identity"]["collections"])
        self.index_embedder = dict(self._checkpoint["identity"]["embedder"])
        self.max_length = int(
            max_length
            if max_length is not None
            else self.index_embedder.get("max_length", 1024)
        )
        if self._embedder is not None:
            self._validate_embedder(self._embedder)

    def _load_checkpoint(self) -> dict:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"P4 checkpoint 不存在：{self.checkpoint_path}")
        checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("state") != "complete":
            raise VectorIndexError("P4 向量索引尚未完成")
        if not checkpoint.get("identity"):
            raise VectorIndexError("P4 checkpoint 缺少 identity")
        return checkpoint

    def _validate_embedder(self, embedder) -> None:
        info = embedder.info()
        expected = self.index_embedder
        for key in ("model_id", "dimension"):
            if info.get(key) != expected.get(key):
                raise VectorIndexError(
                    f"查询模型与索引不一致：{key}={info.get(key)} "
                    f"!= {expected.get(key)}"
                )

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = BGEM3Embedder(
                model_path=self.model_cache,
                model_id=self.index_embedder.get("model_id", "BAAI/bge-m3"),
                revision=self.index_embedder.get("revision"),
                device=self.device,
                max_length=self.max_length,
                batch_size=self.batch_size,
            )
            self._validate_embedder(self._embedder)
        return self._embedder

    @property
    def client(self):
        if self._client is None:
            self._client = create_client(self.store_dir)
        return self._client

    def search(
        self,
        query: str,
        top_k_per_collection: int = DEFAULT_COLLECTION_TOP_K,
        entity_types: Iterable[str] | None = None,
    ) -> list[DocumentHit]:
        requested = list(entity_types or ENTITY_TYPES)
        query_embedding = self.embedder.encode_queries([query])[0]
        merged: list[DocumentHit] = []
        for entity_type in requested:
            collection_name = self.collections.get(entity_type)
            if not collection_name:
                continue
            try:
                collection = self.client.get_collection(collection_name)
            except Exception as exc:  # noqa: BLE001 - collection may be absent
                raise VectorIndexError(
                    f"P4 collection 不存在：{collection_name}"
                ) from exc
            rows = query_collection(
                collection, query_embedding, max(1, int(top_k_per_collection))
            )
            for collection_rank, row in enumerate(rows, 1):
                metadata = row.get("metadata") or {}
                distance = float(row.get("distance") or 0.0)
                merged.append(
                    DocumentHit(
                        doc_id=str(row["doc_id"]),
                        entity_type=str(metadata.get("entity_type", entity_type)),
                        entity_id=int(metadata.get("entity_id", -1)),
                        subject_id=int(metadata.get("subject_id", -1)),
                        text=str(row.get("document") or ""),
                        metadata=metadata,
                        score=1.0 - distance,
                        rank=0,
                        source="vector",
                        snippet=str(row.get("document") or "")[:160],
                        collection_rank=collection_rank,
                    )
                )
        merged.sort(key=lambda hit: (-hit.score, hit.doc_id))
        for rank, hit in enumerate(merged, 1):
            hit.rank = rank
        return merged

    def search_timed(
        self,
        query: str,
        top_k_per_collection: int = DEFAULT_COLLECTION_TOP_K,
        entity_types: Iterable[str] | None = None,
    ) -> tuple[list[DocumentHit], float]:
        started = time.perf_counter()
        hits = self.search(query, top_k_per_collection, entity_types)
        return hits, (time.perf_counter() - started) * 1000.0

    def close(self) -> None:
        if self._client is not None:
            self._client = None
        if self._owns_embedder and self._embedder is not None:
            close = getattr(self._embedder, "close", None)
            if close is not None:
                close()
            self._embedder = None

    def __enter__(self) -> "VectorSearcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
