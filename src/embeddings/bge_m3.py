"""BGE-M3 embedding backend.

The same CLS pooling and L2 normalization are used for index documents and
queries.  BGE-M3 does not require different instructions for the two sides, so
the only query/document difference is the caller's batching policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DEFAULT_MODEL_ID = "BAAI/bge-m3"
DEFAULT_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DEFAULT_MAX_LENGTH = 1024


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def read_model_manifest(model_path: Path) -> dict[str, Any]:
    manifest_path = model_path / "plotseeker-model.json"
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


class BGEM3Embedder:
    def __init__(
        self,
        model_path: str | Path,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        device: str = "auto",
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = 16,
        use_fp16: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"模型目录不存在：{self.model_path}。先运行 scripts/download_model.py。"
            )

        self.model_id = model_id
        manifest = read_model_manifest(self.model_path)
        self.revision = revision or manifest.get("revision") or DEFAULT_REVISION
        self.device = _resolve_device(device)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.use_fp16 = bool(use_fp16 and self.device == "cuda")
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": False,
            "token": False,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), **load_kwargs
        )
        self.model = AutoModel.from_pretrained(str(self.model_path), **load_kwargs)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        self._dimension = int(self.model.config.hidden_size)

    @property
    def dimension(self) -> int:
        return self._dimension

    def info(self) -> dict[str, Any]:
        manifest = read_model_manifest(self.model_path)
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "dimension": self.dimension,
            "max_length": self.max_length,
            "pooling": "cls",
            "normalize": True,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "batch_size": self.batch_size,
            "backend": "transformers",
            "model_path": str(self.model_path.resolve()),
            "model_manifest_endpoint": manifest.get("endpoint"),
            "model_manifest_downloaded_at": manifest.get("downloaded_at"),
        }

    def _encode(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        size = int(batch_size or self.batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive")

        torch = self._torch
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), size):
                batch = list(texts[start : start + size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask", "token_type_ids"}
                }
                hidden = self.model(**encoded).last_hidden_state
                pooled = hidden[:, 0]
                normalized = torch.nn.functional.normalize(
                    pooled.float(), p=2, dim=1
                )
                outputs.append(normalized.cpu().numpy().astype(np.float32))
        return np.vstack(outputs)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def close(self) -> None:
        del self.model
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
