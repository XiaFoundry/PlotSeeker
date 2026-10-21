#!/usr/bin/env python
"""Build the P4 Chroma vector index from P3 retrieval documents.

Examples
--------
Build the default deterministic 10,000-document index with BGE-M3::

    python scripts/build_index.py

Resume after an interrupted build (same command)::

    python scripts/build_index.py

Explicitly replace an existing index::

    python scripts/build_index.py --rebuild

Offline smoke build without model weights::

    python scripts/build_index.py --fake-embedder --sample-sizes 20,20,20 \
        --store-dir storage/vector_store_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    DOCUMENTS_DIR,
    INDEX_CHECKPOINT_PATH,
    INDEX_REPORT_PATH,
    MODEL_CACHE_DIR,
    VECTOR_STORE_DIR,
)
from embeddings.bge_m3 import (  # noqa: E402
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    BGEM3Embedder,
)
from embeddings.chroma_store import IndexConflict  # noqa: E402
from embeddings.fake import DeterministicFakeEmbedder  # noqa: E402
from embeddings.indexer import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_SAMPLE_SIZES,
    DEFAULT_SANITY_QUERIES,
    ENTITY_TYPES,
    IndexAlreadyComplete,
    IndexInterrupted,
    SourceChanged,
    run_build_index,
)


def _parse_sample_sizes(value: str) -> dict[str, int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != len(ENTITY_TYPES):
        raise argparse.ArgumentTypeError(
            "sample sizes must be three comma-separated integers, "
            "e.g. 4000,3000,3000"
        )
    try:
        sizes = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample sizes must be integers") from exc
    if any(size < 0 for size in sizes):
        raise argparse.ArgumentTypeError("sample sizes must be non-negative")
    return dict(zip(ENTITY_TYPES, sizes))


def _command_string(
    args: argparse.Namespace, checkpoint_path: Path, report_path: Path
) -> str:
    parts = ["python", "scripts/build_index.py"]
    if Path(args.manifest) != DOCUMENTS_DIR / "manifest.json":
        parts.extend(["--manifest", str(args.manifest)])
    if Path(args.store_dir) != VECTOR_STORE_DIR:
        parts.extend(["--store-dir", str(args.store_dir)])
    if checkpoint_path != INDEX_CHECKPOINT_PATH:
        parts.extend(["--checkpoint", str(checkpoint_path)])
    if report_path != INDEX_REPORT_PATH:
        parts.extend(["--report", str(report_path)])
    if args.sample_sizes != DEFAULT_SAMPLE_SIZES:
        parts.extend(
            [
                "--sample-sizes",
                ",".join(str(args.sample_sizes[item]) for item in ENTITY_TYPES),
            ]
        )
    if args.seed != 42:
        parts.extend(["--seed", str(args.seed)])
    if args.batch_size != DEFAULT_BATCH_SIZE:
        parts.extend(["--batch-size", str(args.batch_size)])
    if args.max_length != DEFAULT_MAX_LENGTH:
        parts.extend(["--max-length", str(args.max_length)])
    if args.device != "auto":
        parts.extend(["--device", args.device])
    if args.fake_embedder:
        parts.append("--fake-embedder")
    if args.rebuild:
        parts.append("--rebuild")
    if not args.verify_source:
        parts.append("--no-verify-source")
    if args.sanity_queries != DEFAULT_SANITY_QUERIES:
        parts.extend(["--sanity-queries", str(args.sanity_queries)])
    if args.stop_after_batches is not None:
        parts.extend(["--stop-after-batches", str(args.stop_after_batches)])
    return " ".join(parts)


def _make_embedder(args: argparse.Namespace):
    if args.fake_embedder:
        return DeterministicFakeEmbedder(dimension=args.fake_dimension)
    return BGEM3Embedder(
        model_path=args.model_cache,
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a persistent Chroma index from P3 documents."
    )
    parser.add_argument(
        "--manifest",
        default=str(DOCUMENTS_DIR / "manifest.json"),
        help="P3 manifest.json path.",
    )
    parser.add_argument(
        "--store-dir",
        default=str(VECTOR_STORE_DIR),
        help="Chroma persistence directory.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint JSON path (default: <store-dir>/p4_checkpoint.json).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Index report JSON path (default: <store-dir>/p4_report.json).",
    )
    parser.add_argument(
        "--model-cache",
        default=str(MODEL_CACHE_DIR / "bge-m3"),
        help="Local BGE-M3 snapshot directory.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Model id.")
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION, help="Pinned model revision."
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto", help="Device."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Tokenizer max length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Documents encoded and written per batch.",
    )
    parser.add_argument(
        "--sample-sizes",
        type=_parse_sample_sizes,
        default=dict(DEFAULT_SAMPLE_SIZES),
        help="Deterministic sample sizes: subject,character,episode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--sanity-queries",
        type=int,
        default=DEFAULT_SANITY_QUERIES,
        help="Number of known-target sanity queries (0 disables).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Explicitly delete target collections and rebuild from scratch.",
    )
    parser.add_argument(
        "--no-verify-source",
        dest="verify_source",
        action="store_false",
        help="Skip SHA256 verification of P3 JSONL files.",
    )
    parser.add_argument(
        "--fake-embedder",
        action="store_true",
        help="Use the deterministic fake backend (no model download).",
    )
    parser.add_argument(
        "--fake-dimension",
        type=int,
        default=64,
        help="Dimension for --fake-embedder.",
    )
    parser.add_argument(
        "--stop-after-batches",
        type=int,
        default=None,
        help="Debug hook: stop after N newly indexed batches.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the summary.")
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if args.sanity_queries < 0:
        parser.error("--sanity-queries must be non-negative")
    if args.stop_after_batches is not None and args.stop_after_batches <= 0:
        parser.error("--stop-after-batches must be positive")

    store_dir = Path(args.store_dir)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else store_dir / "p4_checkpoint.json"
    )
    report_path = (
        Path(args.report) if args.report else store_dir / "p4_report.json"
    )

    try:
        embedder = _make_embedder(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"嵌入模型加载失败：{exc}", file=sys.stderr)
        return 1

    print(f"嵌入模型：{embedder.info().get('model_id')} ({embedder.dimension} 维)")
    print(f"设备：{embedder.info().get('device')}")

    try:
        report = run_build_index(
            manifest_path=Path(args.manifest),
            store_dir=store_dir,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            embedder=embedder,
            sample_sizes=args.sample_sizes,
            seed=args.seed,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
            verify_source=args.verify_source,
            sanity_queries=args.sanity_queries,
            stop_after_batches=args.stop_after_batches,
            command=_command_string(args, checkpoint_path, report_path),
            verbose=not args.quiet,
        )
    except IndexAlreadyComplete as exc:
        print(f"索引已存在：{exc}", file=sys.stderr)
        return 2
    except IndexConflict as exc:
        print(f"拒绝覆盖：{exc}", file=sys.stderr)
        return 2
    except IndexInterrupted as exc:
        print(f"索引构建已中断：{exc}", file=sys.stderr)
        print(f"checkpoint：{checkpoint_path}")
        return 3
    except SourceChanged as exc:
        print(f"源文档已变化：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"索引构建失败：{exc}", file=sys.stderr)
        return 1
    finally:
        close = getattr(embedder, "close", None)
        if close is not None:
            close()

    print(f"索引目录：{report['index']['store_dir']}")
    print(f"报告：{report['_report_path']}")
    for entity_type in ENTITY_TYPES:
        info = report["documents"][entity_type]
        print(
            f"  {info['collection']:<20} {info['reopened_count']:>7,} 条"
            f"  抽样 {info['sampled']:>7,}"
        )
    performance = report["performance"]
    print(
        f"编码速度：{performance['documents_per_sec_encoding']} docs/s；"
        f"总耗时 {performance['elapsed_sec']}s"
    )
    print(
        f"峰值内存：{performance['peak_rss_mb']} MiB；"
        f"GPU：{performance['gpu']}"
    )
    sanity = report["sanity"]
    print(
        f"Sanity：{sanity['hit_at_5']}/{sanity['query_count']} hit@5，"
        f"{sanity['hit_at_1']}/{sanity['query_count']} hit@1；"
        f"P50 {sanity['latency_ms']['p50']}ms / "
        f"P95 {sanity['latency_ms']['p95']}ms"
    )
    print(f"索引体积：{report['index']['size_bytes'] / 1024 / 1024:.1f} MiB")
    if report["checkpoint"]["resumed"]:
        print("本次为断点恢复。")
    if report["rebuild"]["performed"]:
        print("本次为显式重建。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
