#!/usr/bin/env python
"""Command-line entry point for the Bangumi Archive data audit.

Examples
--------
Full audit (writes ``docs/data-audit.md``)::

    python scripts/audit_data.py

Smoke audit on the first 5,000 lines of each file::

    python scripts/audit_data.py --limit 5000 --output scratch/data-audit-smoke.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DOCS_DIR, RAW_DIR  # noqa: E402
from ingestion.audit import run_audit  # noqa: E402


def _build_command(args: argparse.Namespace) -> str:
    parts = ["python", "scripts/audit_data.py"]
    if Path(args.raw_dir) != RAW_DIR:
        parts.extend(["--raw-dir", str(args.raw_dir)])
    if Path(args.output) != DOCS_DIR / "data-audit.md":
        parts.extend(["--output", str(args.output)])
    if args.limit is not None:
        parts.extend(["--limit", str(args.limit)])
    if args.sample_size != 3000:
        parts.extend(["--sample-size", str(args.sample_size)])
    if args.seed != 42:
        parts.extend(["--seed", str(args.seed)])
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Streaming audit of the Bangumi Archive JSONL files."
    )
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory containing the 9 JSONL files.")
    parser.add_argument(
        "--output",
        default=str(DOCS_DIR / "data-audit.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Read at most N lines per file (smoke mode; integrity checks are skipped).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="Reservoir sample size for episode-description quality analysis.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.sample_size < 0:
        parser.error("--sample-size must be >= 0")

    try:
        summary = run_audit(
            raw_dir=Path(args.raw_dir),
            output_path=Path(args.output),
            limit=args.limit,
            sample_size=args.sample_size,
            seed=args.seed,
            command=_build_command(args),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"审计失败：{exc}", file=sys.stderr)
        return 1

    print(f"报告：{summary['output']}")
    print(
        "实体记录："
        + ", ".join(f"{name}={count:,}" for name, count in summary["entities"].items())
    )
    print(
        "关联记录："
        + ", ".join(f"{name}={count:,}" for name, count in summary["relations"].items())
    )
    print(f"异常行：{summary['bad_lines']:,}")
    print(f"耗时：{summary['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
