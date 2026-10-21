#!/usr/bin/env python
"""Generate the P6 development benchmark from the P4 sample.

The generated queries are for reproducible development baselines, not a
replacement for independent human evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import BENCHMARKS_DIR, DOCUMENTS_DIR, VECTOR_STORE_DIR  # noqa: E402
from evaluation.benchmark_builder import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    build_benchmark,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate P6 benchmark queries.")
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
        "--output", default=str(DEFAULT_OUTPUT), help="Output JSONL path."
    )
    parser.add_argument(
        "--benchmark-manifest",
        default=str(DEFAULT_MANIFEST),
        help="Benchmark metadata JSON path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Generation seed.")
    args = parser.parse_args(argv)

    try:
        manifest = build_benchmark(
            manifest_path=Path(args.manifest),
            p4_checkpoint_path=Path(args.p4_checkpoint),
            output_path=Path(args.output),
            manifest_output=Path(args.benchmark_manifest),
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"评测集生成失败：{exc}", file=sys.stderr)
        return 1

    print(f"输出：{Path(args.output)}")
    print(f"清单：{Path(args.benchmark_manifest)}")
    print(
        f"查询数：{manifest['query_count']}  "
        f"可回答：{manifest['answerable_count']}  "
        f"不可回答：{manifest['unanswerable_count']}"
    )
    for clue_type, count in manifest["clue_type_counts"].items():
        print(f"  {clue_type:<22} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
