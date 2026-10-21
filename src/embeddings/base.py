"""Embedding backend contract shared by index builds and query-time search."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    """Minimal interface required by the P4 index builder.

    Documents and queries must be encoded with the same backend and the same
    preprocessing rules. Implementations return L2-normalized ``float32``
    matrices with shape ``(len(texts), dimension)``.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str | None: ...

    @property
    def dimension(self) -> int: ...

    @property
    def max_length(self) -> int: ...

    def info(self) -> dict[str, Any]: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...
