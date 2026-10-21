#!/usr/bin/env python
"""Run P6 keyword / vector / hybrid baselines on the generated benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    BENCHMARKS_DIR,
    DOCS_DIR,
    MODEL_CACHE_DIR,
    SQLITE_PATH,
    STORAGE_DIR,
    VECTOR_STORE_DIR,
)
from embeddings.bge_m3 import BGEM3Embedder  # noqa: E402
from embeddings.fake import DeterministicFakeEmbedder  # noqa: E402
from evaluation.runner import run_evaluation  # noqa: E402
from retrieval.keyword_index import KEYWORD_DB_PATH  # noqa: E402
from retrieval.search import MODES, RetrievalService  # noqa: E402

DEFAULT_QUERIES = BENCHMARKS_DIR / "queries.jsonl"
DEFAULT_REPORT = STORAGE_DIR / "evaluation" / "p6_report.json"
DEFAULT_MARKDOWN = DOCS_DIR / "evaluation.md"
DEFAULT_FAILURES = BENCHMARKS_DIR / "failures.md"


def _command_string(args: argparse.Namespace) -> str:
    parts = ["python", "scripts/evaluate.py"]
    if Path(args.queries) != DEFAULT_QUERIES:
        parts.extend(["--queries", str(args.queries)])
    if Path(args.keyword_db) != KEYWORD_DB_PATH:
        parts.extend(["--keyword-db", str(args.keyword_db)])
    if Path(args.store_dir) != VECTOR_STORE_DIR:
        parts.extend(["--store-dir", str(args.store_dir)])
    if args.top_k != 10:
        parts.extend(["--top-k", str(args.top_k)])
    if args.candidate_k != 50:
        parts.extend(["--candidate-k", str(args.candidate_k)])
    if args.modes != ",".join(MODES):
        parts.extend(["--modes", args.modes])
    if args.device != "auto":
        parts.extend(["--device", args.device])
    if args.fake_embedder:
        parts.append("--fake-embedder")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run P6 retrieval baselines.")
    parser.add_argument(
        "--queries", default=str(DEFAULT_QUERIES), help="Benchmark JSONL path."
    )
    parser.add_argument(
        "--keyword-db", default=str(KEYWORD_DB_PATH), help="P5 FTS5 database."
    )
    parser.add_argument(
        "--store-dir", default=str(VECTOR_STORE_DIR), help="P4 Chroma directory."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(VECTOR_STORE_DIR / "p4_checkpoint.json"),
        help="P4 checkpoint path.",
    )
    parser.add_argument(
        "--model-cache",
        default=str(MODEL_CACHE_DIR / "bge-m3"),
        help="BGE-M3 snapshot directory.",
    )
    parser.add_argument(
        "--sqlite", default=str(SQLITE_PATH), help="P2 SQLite database."
    )
    parser.add_argument(
        "--modes", default=",".join(MODES), help="Comma-separated modes."
    )
    parser.add_argument("--top-k", type=int, default=10, help="Subjects returned.")
    parser.add_argument(
        "--candidate-k", type=int, default=50, help="Documents per channel."
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto", help="Device."
    )
    parser.add_argument(
        "--report", default=str(DEFAULT_REPORT), help="JSON report path."
    )
    parser.add_argument(
        "--markdown", default=str(DEFAULT_MARKDOWN), help="Markdown report path."
    )
    parser.add_argument(
        "--failures", default=str(DEFAULT_FAILURES), help="Failure log path."
    )
    parser.add_argument(
        "--fake-embedder",
        action="store_true",
        help="Use the deterministic fake embedder (smoke tests only).",
    )
    parser.add_argument(
        "--fake-dimension", type=int, default=16, help="Fake embedding dimension."
    )
    args = parser.parse_args(argv)

    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    if not modes or any(mode not in MODES for mode in modes):
        parser.error(f"--modes must be a subset of {','.join(MODES)}")
    if args.top_k <= 0 or args.candidate_k <= 0:
        parser.error("--top-k and --candidate-k must be positive")

    if not Path(args.keyword_db).is_file():
        print(
            f"关键词索引不存在：{args.keyword_db}；先运行 scripts/build_keyword_index.py",
            file=sys.stderr,
        )
        return 1

    embedder = None
    model_info = {}
    load_started = time.perf_counter()
    if args.fake_embedder:
        embedder = DeterministicFakeEmbedder(args.fake_dimension)
        model_info = embedder.info()
    else:
        try:
            embedder = BGEM3Embedder(
                model_path=Path(args.model_cache),
                device=args.device,
                max_length=1024,
                batch_size=16,
            )
            model_info = embedder.info()
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"Embedding 模型加载失败：{exc}", file=sys.stderr)
            return 1
    model_info["load_sec"] = round(time.perf_counter() - load_started, 2)

    service = RetrievalService(
        keyword_db=Path(args.keyword_db),
        store_dir=Path(args.store_dir),
        checkpoint_path=Path(args.checkpoint),
        model_cache=Path(args.model_cache),
        sqlite_path=Path(args.sqlite),
        vector_embedder=embedder,
        device=args.device,
    )
    try:
        report = run_evaluation(
            queries_path=Path(args.queries),
            service=service,
            model_info=model_info,
            command=_command_string(args),
            modes=modes,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            report_path=Path(args.report),
            markdown_path=Path(args.markdown),
            failures_path=Path(args.failures),
            keyword_db_path=Path(args.keyword_db),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"评测失败：{exc}", file=sys.stderr)
        return 1
    finally:
        service.close()
        if embedder is not None and not args.fake_embedder:
            close = getattr(embedder, "close", None)
            if close is not None:
                close()

    print(f"报告：{Path(args.report)}")
    print(f"Markdown：{Path(args.markdown)}")
    print(f"失败记录：{Path(args.failures)}")
    for mode in modes:
        summary = report["summary"][mode]
        print(
            f"  {mode:<7} R@5={summary['recall_at_5']} "
            f"MRR@10={summary['mrr_at_10']} Top1={summary['top1_accuracy']} "
            f"P50={summary['latency_ms']['p50']}ms "
            f"P95={summary['latency_ms']['p95']}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
