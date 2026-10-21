#!/usr/bin/env python
"""Command-line demo for P5 keyword, vector and hybrid retrieval.

Examples
--------
Hybrid search::

    python scripts/search.py "失去记忆的魔王在现代学校"

Keyword only::

    python scripts/search.py "时间循环" --mode keyword --top-k 5

Machine-readable output::

    python scripts/search.py "妹妹用脚抵住门" --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import MODEL_CACHE_DIR, SQLITE_PATH, VECTOR_STORE_DIR  # noqa: E402
from retrieval.keyword_index import KEYWORD_DB_PATH  # noqa: E402
from retrieval.search import MODES, RetrievalService  # noqa: E402


def _print_human(result: dict) -> None:
    print(f"查询：{result['query']}")
    print(
        f"模式：{result['mode']}  confidence={result['confidence']}  "
        f"candidates={result['counts']['subject_candidates']}"
    )
    if result.get("warnings"):
        print(f"警告：{', '.join(result['warnings'])}")
    if result.get("vector_top_similarity") is not None:
        print(f"向量最高相似度：{result['vector_top_similarity']}")
    print(
        f"耗时：total={result['timings_ms']['total']}ms "
        f"keyword={result['timings_ms']['keyword']}ms "
        f"vector={result['timings_ms']['vector']}ms"
    )
    if not result["candidates"]:
        print("未找到可追溯的作品候选。")
        return
    for index, candidate in enumerate(result["candidates"], 1):
        names = candidate["name_cn"] or candidate["name"]
        print(
            f"\n{index}. subject:{candidate['subject_id']}  {names}  "
            f"score={candidate['score']:.6f}"
        )
        if candidate["name"] and candidate["name"] != names:
            print(f"   原名：{candidate['name']}")
        ranks = []
        if candidate["keyword_rank"] is not None:
            ranks.append(f"keyword#{candidate['keyword_rank']}")
        if candidate["vector_rank"] is not None:
            ranks.append(f"vector#{candidate['vector_rank']}")
        if ranks:
            print(f"   排名：{'  '.join(ranks)}")
        for evidence in candidate["evidence"]:
            snippet = evidence["snippet"].replace("\n", " ")
            print(
                f"   - [{evidence['source']}] {evidence['doc_id']} "
                f"rank={evidence['rank']} path={evidence['relation_path']}"
            )
            if snippet:
                print(f"     {snippet}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P5 retrieval command-line demo.")
    parser.add_argument("query", nargs="?", help="Natural-language query.")
    parser.add_argument("--query", dest="query_option", help="Natural-language query.")
    parser.add_argument("--mode", choices=MODES, default="hybrid", help="Retrieval mode.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of subjects.")
    parser.add_argument(
        "--candidate-k", type=int, default=50, help="Documents fetched per channel."
    )
    parser.add_argument(
        "--rrf-k", type=int, default=60, help="RRF smoothing constant."
    )
    parser.add_argument(
        "--keyword-db", default=str(KEYWORD_DB_PATH), help="P5 FTS5 database."
    )
    parser.add_argument(
        "--store-dir", default=str(VECTOR_STORE_DIR), help="P4 Chroma directory."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="P4 checkpoint path (default: <store-dir>/p4_checkpoint.json).",
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
        "--device", choices=("auto", "cuda", "cpu"), default="auto", help="Device."
    )
    parser.add_argument(
        "--tokenizer", choices=("jieba", "bigram"), default=None, help="FTS tokenizer."
    )
    parser.add_argument("--json", action="store_true", help="Print JSON.")
    args = parser.parse_args(argv)

    query = args.query_option or args.query
    if query is None or not query.strip():
        parser.error("query is required")
    if args.top_k <= 0 or args.candidate_k <= 0:
        parser.error("--top-k and --candidate-k must be positive")

    service = RetrievalService(
        keyword_db=Path(args.keyword_db),
        store_dir=Path(args.store_dir),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        model_cache=Path(args.model_cache),
        sqlite_path=Path(args.sqlite),
        device=args.device,
        keyword_tokenizer=args.tokenizer,
        rrf_k=args.rrf_k,
    )
    try:
        result = service.search(
            query, mode=args.mode, top_k=args.top_k, candidate_k=args.candidate_k
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"检索失败：{exc}", file=sys.stderr)
        return 1
    finally:
        service.close()

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
