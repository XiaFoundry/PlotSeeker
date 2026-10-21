"""Deterministic embedding backend used by tests and offline smoke builds."""

from __future__ import annotations

import hashlib

import numpy as np


class DeterministicFakeEmbedder:
    """Hash-based vectors with stable output across processes and machines."""

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return "fake-deterministic-v1"

    @property
    def revision(self) -> str | None:
        return "1"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_length(self) -> int:
        return 8192

    def info(self) -> dict:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "dimension": self.dimension,
            "max_length": self.max_length,
            "pooling": "hash",
            "normalize": True,
            "device": "cpu",
            "dtype": "float32",
            "backend": "deterministic-fake",
        }

    def _encode(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        vector = np.random.default_rng(seed).standard_normal(self._dimension)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            norm = 1.0
        return (vector / norm).astype(np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        return np.vstack([self._encode(text) for text in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_documents(texts)
