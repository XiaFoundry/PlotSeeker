#!/usr/bin/env python
"""Compare direct hybrid retrieval with the P9 agent on the P6 benchmark."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.agent import PlotSeekerAgent  # noqa: E402
from agent.evaluation import run_agent_evaluation  # noqa: E402
from agent.tools import AgentTools  # noqa: E402
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
from retrieval.keyword_index import KEYWORD_DB_PATH  # noqa: E402

DEFAULT_QUERIES = BENCHMARKS_DIR / "queries.jsonl"
DEFAULT_REPORT = STORAGE_DIR / "evaluation" / "p9_agent_report.json"
DEFAULT_MARKDOWN = DOCS_DIR / "agent-evaluation.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P9 agent comparison evaluation.")
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
        "--device", choices=("auto", "cuda", "cpu"), default="auto", help="Device."
    )
    parser.add_argument("--top-k", type=int, default=10, help="Reported candidates.")
    parser.add_argument(
        "--candidate-k", type=int, default=50, help="Baseline document candidates."
    )
    parser.add_argument(
        "--report", default=str(DEFAULT_REPORT), help="JSON report path."
    )
    parser.add_argument(
        "--markdown", default=str(DEFAULT_MARKDOWN), help="Markdown report path."
    )
    parser.add_argument(
        "--fake-embedder",
        action="store_true",
        help="Use the deterministic fake embedder (smoke only).",
    )
    args = parser.parse_args(argv)

    if not Path(args.keyword_db).is_file():
        print(f"关键词索引不存在：{args.keyword_db}", file=sys.stderr)
        return 1

    embedder = None
    try:
        if args.fake_embedder:
            embedder = DeterministicFakeEmbedder(16)
        else:
            embedder = BGEM3Embedder(
                model_path=Path(args.model_cache),
                device=args.device,
                max_length=1024,
                batch_size=16,
            )
        tools = AgentTools(
            keyword_db=Path(args.keyword_db),
            store_dir=Path(args.store_dir),
            checkpoint_path=Path(args.checkpoint),
            model_cache=Path(args.model_cache),
            sqlite_path=Path(args.sqlite),
            embedder=embedder,
            device=args.device,
        )
        agents = {
            "agent": PlotSeekerAgent(tools, max_steps=4, top_k=10),
            "agent_clarify_low": PlotSeekerAgent(
                tools, max_steps=4, top_k=10, clarify_on_low=True
            ),
        }
        try:
            tools.retrieval.search(
                "时间循环", mode="hybrid", top_k=1, candidate_k=5
            )
            for agent in agents.values():
                agent.run("时间循环", top_k=1)
            report = run_agent_evaluation(
                queries_path=Path(args.queries),
                baseline_service=tools.retrieval,
                tools=tools,
                agents=agents,
                model_info=embedder.info(),
                report_path=Path(args.report),
                markdown_path=Path(args.markdown),
                command="python scripts/evaluate_agent.py",
                top_k=args.top_k,
                candidate_k=args.candidate_k,
            )
        finally:
            tools.close()
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"Agent 评测失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if embedder is not None and not args.fake_embedder:
            close = getattr(embedder, "close", None)
            if close is not None:
                close()

    print(f"报告：{Path(args.report)}")
    print(f"Markdown：{Path(args.markdown)}")
    print(f"结论：{report['conclusion']}")
    for name, entry in report["summary"].items():
        print(
            f"  {name:<18} R@5={entry['recall_at_5']} "
            f"MRR@10={entry['mrr_at_10']} Top1={entry['top1_accuracy']} "
            f"clarify={entry['answerable_clarification_rate']} "
            f"P50={entry['latency_ms']['p50']}ms "
            f"tools={entry['avg_tool_calls']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
