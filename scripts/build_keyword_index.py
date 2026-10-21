#!/usr/bin/env python
"""Build the P5 SQLite FTS5 keyword index from the P4 document sample.

The document IDs are read from the completed P4 checkpoint, so keyword
retrieval and vector retrieval cover exactly the same 10,000 documents.

Examples
--------
Build the default index (jieba and bigram tables + comparison report)::

    python scripts/build_keyword_index.py

Explicitly rebuild::

    python scripts/build_keyword_index.py --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DOCUMENTS_DIR, VECTOR_STORE_DIR  # noqa: E402
from retrieval.keyword_index import (  # noqa: E402
    KEYWORD_DB_PATH,
    KEYWORD_REPORT_PATH,
    build_keyword_index,
)
from retrieval.preprocess import TOKENIZERS  # noqa: E402


def _parse_tokenizers(value: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [item for item in items if item not in TOKENIZERS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown tokenizers: {', '.join(unknown)}")
    if not items:
        raise argparse.ArgumentTypeError("at least one tokenizer is required")
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the P5 FTS5 keyword index from the P4 sample."
    )
    parser.add_argument(
        "--manifest",
        default=str(DOCUMENTS_DIR / "manifest.json"),
        help="P3 manifest.json path.",
    )
    parser.add_argument(
        "--p4-checkpoint",
        default=str(VECTOR_STORE_DIR / "p4_checkpoint.json"),
        help="Completed P4 checkpoint path.",
    )
    parser.add_argument(
        "--output",
        default=str(KEYWORD_DB_PATH),
        help="Output SQLite FTS5 database.",
    )
    parser.add_argument(
        "--report",
        default=str(KEYWORD_REPORT_PATH),
        help="Build report JSON path.",
    )
    parser.add_argument(
        "--tokenizers",
        type=_parse_tokenizers,
        default=TOKENIZERS,
        help="Comma-separated tokenizers to index: jieba,bigram.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Explicitly replace an existing keyword index.",
    )
    parser.add_argument(
        "--no-verify-source",
        dest="verify_source",
        action="store_false",
        help="Skip P3 SHA256 verification (not recommended).",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the summary.")
    args = parser.parse_args(argv)

    try:
        report = build_keyword_index(
            manifest_path=Path(args.manifest),
            p4_checkpoint_path=Path(args.p4_checkpoint),
            output_path=Path(args.output),
            report_path=Path(args.report),
            tokenizers=args.tokenizers,
            rebuild=args.rebuild,
            verify_source=args.verify_source,
            verbose=not args.quiet,
        )
    except FileExistsError as exc:
        print(f"拒绝覆盖：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"关键词索引构建失败：{exc}", file=sys.stderr)
        return 1

    print(f"索引文件：{report['output_path']}")
    print(f"报告：{Path(args.report)}")
    print(f"文档数：{report['document_count']:,}")
    print(f"默认分词：{report['default_tokenizer']}")
    for name, metrics in report["tokenizer_comparison"].items():
        print(
            f"  {name:<7} recall@5={metrics['recall_at_5']} "
            f"MRR@10={metrics['mrr_at_10']} queries={metrics['queries']}"
        )
    print(f"耗时：{report['elapsed_sec']}s  体积：{report['size_bytes'] / 1024 / 1024:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
