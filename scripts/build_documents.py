#!/usr/bin/env python
"""Build P3 retrieval documents from the SQLite database.

Examples
--------
Full build into ``data/processed/documents``::

    python scripts/build_documents.py

Rebuild existing outputs::

    python scripts/build_documents.py --overwrite

Smoke build (at most 50 documents per entity type)::

    python scripts/build_documents.py --limit 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DOCUMENTS_DIR, SQLITE_PATH  # noqa: E402
from ingestion.document_builder import run_build  # noqa: E402


def _build_command(args: argparse.Namespace) -> str:
    parts = ["python", "scripts/build_documents.py"]
    if Path(args.db) != SQLITE_PATH:
        parts.extend(["--db", str(args.db)])
    if Path(args.output_dir) != DOCUMENTS_DIR:
        parts.extend(["--output-dir", str(args.output_dir)])
    if args.limit is not None:
        parts.extend(["--limit", str(args.limit)])
    if args.seed != 42:
        parts.extend(["--seed", str(args.seed)])
    if args.overwrite:
        parts.append("--overwrite")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build subject / character / episode retrieval documents from SQLite."
    )
    parser.add_argument("--db", default=str(SQLITE_PATH), help="SQLite database path.")
    parser.add_argument(
        "--output-dir",
        default=str(DOCUMENTS_DIR),
        help="Directory for JSONL documents and manifest.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Write at most N documents per entity type (smoke mode).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing document files. New files are built before replacement.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary.")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    try:
        summary = run_build(
            db_path=Path(args.db),
            output_dir=Path(args.output_dir),
            limit=args.limit,
            overwrite=args.overwrite,
            seed=args.seed,
            command=_build_command(args),
            verbose=not args.quiet,
        )
    except FileExistsError as exc:
        print(f"拒绝覆盖：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"文档构建失败：{exc}", file=sys.stderr)
        return 1

    print(f"输出目录：{summary['output_dir']}")
    print(f"清单：{summary['manifest_path']}")
    for entity_type, info in summary["files"].items():
        print(f"  {entity_type:<10} {info['lines']:>9,} 篇  {info['bytes'] / 1024 / 1024:>8.1f} MiB")
    for entity_type, counts in summary["counts"].items():
        filtered = counts["filtered_total"]
        print(f"  {entity_type:<10} 源 {counts['source_total']:>9,}  过滤 {filtered:>9,}")
    if args.limit is not None:
        print("提示：限量模式的过滤计数仅覆盖已扫描部分，不代表全量。")
    traceability = summary["traceability"]
    print(f"溯源抽样：{traceability['verified']}/{traceability['sample_size']}")
    print(f"耗时：{summary['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
