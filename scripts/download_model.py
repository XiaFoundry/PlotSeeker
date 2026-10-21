#!/usr/bin/env python
"""Download an embedding model snapshot from a HuggingFace mirror.

The download is pinned to an explicit revision and writes a small manifest so
index builds can record exactly which weights were used.  The implicit
HuggingFace token is deliberately disabled: project caches must not depend on
the developer's stored credentials.

Examples
--------
Default BGE-M3 download through ``hf-mirror.com``::

    python scripts/download_model.py

Custom mirror and output directory::

    python scripts/download_model.py --endpoint https://hf-mirror.com \
        --output storage/model_cache/bge-m3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import MODEL_CACHE_DIR  # noqa: E402

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_FALLBACK_ENDPOINT = "https://huggingface.co"

# Only the files required by transformers' AutoModel / AutoTokenizer are
# downloaded.  The ONNX exports and documentation images are not needed.
ALLOW_PATTERNS = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "pytorch_model.bin",
    "1_Pooling/config.json",
]


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download a pinned embedding model snapshot from a mirror."
    )
    parser.add_argument("--repo", default=DEFAULT_MODEL, help="HuggingFace repo id.")
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Pinned revision/commit. Use 'main' to resolve dynamically.",
    )
    parser.add_argument(
        "--endpoint", default=DEFAULT_ENDPOINT, help="HuggingFace-compatible mirror."
    )
    parser.add_argument(
        "--fallback-endpoint",
        default=DEFAULT_FALLBACK_ENDPOINT,
        help="Endpoint to try when the mirror redirects or is unavailable.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Only use --endpoint.",
    )
    parser.add_argument(
        "--output",
        default=str(MODEL_CACHE_DIR / "bge-m3"),
        help="Local model directory.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Concurrent download workers."
    )
    args = parser.parse_args(argv)

    # Must be set before huggingface_hub resolves its configuration.  The
    # implicit token is disabled and xet is skipped because the mirror
    # redirects cross-domain and xet's own log directory may be read-only.
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HOME"] = str(MODEL_CACHE_DIR / "hf_home")

    from huggingface_hub import HfApi, snapshot_download

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoints = [args.endpoint]
    if not args.no_fallback and args.fallback_endpoint not in endpoints:
        endpoints.append(args.fallback_endpoint)

    revision = args.revision
    errors: list[str] = []
    used_endpoint = ""
    for endpoint in endpoints:
        try:
            if revision in {"", "main", "latest"}:
                revision = HfApi(endpoint=endpoint, token=False).model_info(
                    args.repo
                ).sha
            print(f"仓库：{args.repo}")
            print(f"版本：{revision}")
            print(f"端点：{endpoint}")
            print(f"输出：{output_dir}")
            snapshot_download(
                repo_id=args.repo,
                revision=revision,
                local_dir=output_dir,
                token=False,
                allow_patterns=ALLOW_PATTERNS,
                max_workers=args.workers,
                endpoint=endpoint,
            )
            used_endpoint = endpoint
            break
        except Exception as exc:  # noqa: BLE001 - try the next endpoint
            message = f"{endpoint}: {type(exc).__name__}: {exc}"
            errors.append(message)
            print(f"下载失败，尝试下一个端点：{message}", file=sys.stderr)

    if not used_endpoint:
        print("模型下载失败：所有端点均不可用", file=sys.stderr)
        for message in errors:
            print(f"  {message}", file=sys.stderr)
        return 1

    manifest = {
        "repo_id": args.repo,
        "revision": revision,
        "endpoint": used_endpoint,
        "output_dir": str(output_dir.resolve()),
        "allow_patterns": ALLOW_PATTERNS,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "attempt_errors": errors,
    }
    manifest_path = output_dir / "plotseeker-model.json"
    _atomic_write_json(manifest_path, manifest)
    print(f"模型清单：{manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
