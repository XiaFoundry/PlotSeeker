#!/usr/bin/env python
"""Import the Bangumi Archive JSONL files into SQLite.

Examples
--------
Full import (refuses to overwrite an existing database)::

    python scripts/import_sqlite.py

Rebuild the database from scratch::

    python scripts/import_sqlite.py --overwrite

Smoke import (uses a separate database and report by default)::

    python scripts/import_sqlite.py --limit 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import IMPORT_REPORT_PATH, RAW_DIR, SQLITE_PATH, STORAGE_DIR  # noqa: E402
from ingestion.importer import DEFAULT_BATCH_SIZE, run_import  # noqa: E402


def _build_command(args: argparse.Namespace, db_path: Path, report_path: Path) -> str:
    parts = ["python", "scripts/import_sqlite.py"]
    if Path(args.raw_dir) != RAW_DIR:
        parts.extend(["--raw-dir", str(args.raw_dir)])
    if db_path != SQLITE_PATH:
        parts.extend(["--db", str(db_path)])
    if report_path != IMPORT_REPORT_PATH:
        parts.extend(["--report", str(report_path)])
    if args.limit is not None:
        parts.extend(["--limit", str(args.limit)])
    if args.batch_size != DEFAULT_BATCH_SIZE:
        parts.extend(["--batch-size", str(args.batch_size)])
    if args.overwrite:
        parts.append("--overwrite")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import the 9 Bangumi Archive JSONL files into a SQLite database."
    )
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory containing JSONL files.")
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite database path. Defaults to storage/plotseeker.db (or plotseeker-smoke.db with --limit).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Import report JSON path. Defaults to storage/import-report.json (or -smoke).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Read at most N lines per file and only import relations with both endpoints in the limited sets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows per executemany batch (default {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing database. The new database is built in a temp file first.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary.")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

    if args.db:
        db_path = Path(args.db)
    elif args.limit is not None:
        db_path = STORAGE_DIR / "plotseeker-smoke.db"
    else:
        db_path = SQLITE_PATH

    if args.report:
        report_path = Path(args.report)
    elif args.limit is not None:
        report_path = STORAGE_DIR / "import-report-smoke.json"
    else:
        report_path = IMPORT_REPORT_PATH

    try:
        summary = run_import(
            raw_dir=Path(args.raw_dir),
            db_path=db_path,
            report_path=report_path,
            limit=args.limit,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            command=_build_command(args, db_path, report_path),
            verbose=not args.quiet,
        )
    except FileExistsError as exc:
        print(f"拒绝覆盖：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"导入失败：{exc}", file=sys.stderr)
        return 1

    print(f"数据库：{summary['db_path']}")
    print(f"报告：{summary['report_path']}")
    print("表行数：")
    for table, count in summary["table_counts"].items():
        print(f"  {table:<20} {count:>10,}")
    integrity = summary["integrity"]
    print(f"integrity_check：{integrity['pragma_integrity_check']}")
    print(f"foreign_key_violations：{integrity['foreign_key_violations']}")
    bad_lines = sum(
        result["bad_lines"] for result in summary["reconciliation"].values()
    )
    if bad_lines:
        print(f"异常行：{bad_lines:,}", file=sys.stderr)
    bad_tables = [
        table
        for table, result in summary["reconciliation"].items()
        if not result["reconciled"]
    ]
    if bad_tables:
        print("对账未通过：" + ", ".join(bad_tables), file=sys.stderr)
        return 3
    print(f"耗时：{summary['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
