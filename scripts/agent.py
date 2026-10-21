#!/usr/bin/env python
"""P9 single-agent command-line demo.

The agent can route to subject / character / episode search, fetch related
subjects and ask for clarification.  It does not call an LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.agent import PlotSeekerAgent  # noqa: E402
from agent.tools import AgentTools  # noqa: E402
from config import MODEL_CACHE_DIR, SQLITE_PATH, VECTOR_STORE_DIR  # noqa: E402
from embeddings.bge_m3 import BGEM3Embedder  # noqa: E402
from embeddings.fake import DeterministicFakeEmbedder  # noqa: E402
from retrieval.keyword_index import KEYWORD_DB_PATH  # noqa: E402


def _print_human(result: dict) -> None:
    print(f"查询：{result['query']}")
    print(f"回答：{result['answer']}")
    print(f"置信度：{result['confidence']}  耗时：{result['elapsed_ms']}ms")
    if result.get("warnings"):
        print(f"警告：{', '.join(result['warnings'])}")
    print(f"步骤：{' -> '.join(result['steps']) or '-'}")
    if result["tool_calls"]:
        print("工具调用：")
        for call in result["tool_calls"]:
            print(
                f"  - {call['tool']} count={call['count']} "
                f"confidence={call.get('confidence')} "
                f"{call['latency_ms']}ms summary={call.get('summary')}"
            )
    if result.get("clarification"):
        clarification = result["clarification"]
        print(f"需要补充：{clarification['question']}")
        if clarification.get("options"):
            print(f"可选：{' / '.join(clarification['options'])}")
    if result.get("related_subjects"):
        print("相关作品：")
        for item in result["related_subjects"][:10]:
            print(
                f"  - subject:{item['subject_id']} "
                f"{item.get('name_cn') or item.get('name')} "
                f"relation_type={item.get('relation_type')}"
            )
    if result["candidates"]:
        print("候选：")
        for index, candidate in enumerate(result["candidates"], 1):
            name = candidate.get("name_cn") or candidate.get("name")
            print(
                f"  {index}. subject:{candidate['subject_id']} {name} "
                f"score={candidate['score']}"
            )
            for evidence in candidate.get("evidence", [])[:2]:
                snippet = (evidence.get("snippet") or "").replace("\n", " ")[:80]
                print(
                    f"     - [{evidence.get('source')}] {evidence.get('doc_id')} "
                    f"{evidence.get('relation_path')} {snippet}"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P9 single-agent retrieval demo.")
    parser.add_argument("query", nargs="?", help="User query.")
    parser.add_argument("--query", dest="query_option", help="User query.")
    parser.add_argument("--json", action="store_true", help="Print JSON.")
    parser.add_argument("--top-k", type=int, default=5, help="Final candidates.")
    parser.add_argument("--max-steps", type=int, default=4, help="Tool call budget.")
    parser.add_argument(
        "--clarify-low",
        action="store_true",
        help="Ask for clarification instead of returning low-confidence candidates.",
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
    parser.add_argument(
        "--fake-embedder",
        action="store_true",
        help="Use the deterministic fake embedder (smoke only).",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Print tool specs and exit.",
    )
    args = parser.parse_args(argv)

    if args.tools:
        from agent.tools import TOOL_SPECS

        print(json.dumps(TOOL_SPECS, ensure_ascii=False, indent=2))
        return 0

    query = args.query_option or args.query
    if query is None or not query.strip():
        parser.error("query is required")
    if args.top_k <= 0 or args.max_steps <= 0:
        parser.error("--top-k and --max-steps must be positive")

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
        try:
            agent = PlotSeekerAgent(
                tools,
                max_steps=args.max_steps,
                top_k=args.top_k,
                clarify_on_low=args.clarify_low,
            )
            result = agent.run(query, top_k=args.top_k)
        finally:
            tools.close()
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"Agent 运行失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if embedder is not None and not args.fake_embedder:
            close = getattr(embedder, "close", None)
            if close is not None:
                close()

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
