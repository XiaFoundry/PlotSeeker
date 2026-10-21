#!/usr/bin/env python
"""LLM-backed PlotSeeker agent CLI using an OpenAI-compatible API.

Configure ``config/agent.json`` (ignored by Git) or use environment variables:

- PLOTSEEKER_LLM_API_KEY
- PLOTSEEKER_LLM_BASE_URL
- PLOTSEEKER_LLM_MODEL
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.llm_agent import (  # noqa: E402
    LLMAPIError,
    LLMAgent,
    openai_tool_definitions,
)
from agent.llm_config import LLMConfigError, load_llm_config  # noqa: E402
from agent.tools import AgentTools  # noqa: E402
from config import (  # noqa: E402
    AGENT_CONFIG_PATH,
    MODEL_CACHE_DIR,
    SQLITE_PATH,
    VECTOR_STORE_DIR,
)
from embeddings.bge_m3 import BGEM3Embedder  # noqa: E402
from retrieval.keyword_index import KEYWORD_DB_PATH  # noqa: E402


def _print_human(result: dict) -> None:
    print(f"查询：{result['query']}")
    print(f"回答：{result['answer']}")
    print(
        f"模型：{result['model']}  轮次：{result['rounds']}  "
        f"耗时：{result['elapsed_ms']}ms"
    )
    usage = result.get("usage") or {}
    print(
        f"Token：prompt={usage.get('prompt_tokens', 0)} "
        f"completion={usage.get('completion_tokens', 0)} "
        f"total={usage.get('total_tokens', 0)}"
    )
    if result.get("warnings"):
        print(f"警告：{', '.join(result['warnings'])}")
    if result.get("clarification"):
        clarification = result["clarification"]
        print(f"需要补充：{clarification.get('question')}")
        if clarification.get("options"):
            print(f"可选：{' / '.join(clarification['options'])}")
    if result.get("tool_calls"):
        print("工具调用：")
        for call in result["tool_calls"]:
            print(
                f"  - {call['tool']} count={call['count']} "
                f"{call['latency_ms']}ms summary={call.get('summary')} "
                f"error={call.get('error')}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PlotSeeker LLM agent CLI (OpenAI-compatible API)."
    )
    parser.add_argument("query", nargs="?", help="User query.")
    parser.add_argument("--query", dest="query_option", help="User query.")
    parser.add_argument(
        "--config",
        default=str(AGENT_CONFIG_PATH),
        help="Agent JSON config path (default: config/agent.json).",
    )
    parser.add_argument("--api-key", default=None, help="Override API key.")
    parser.add_argument("--base-url", default=None, help="Override API base URL.")
    parser.add_argument("--model", default=None, help="Override model name.")
    parser.add_argument(
        "--max-tool-rounds", type=int, default=None, help="Override tool rounds."
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k.")
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate and print redacted config, then exit.",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Print OpenAI tool definitions, then exit.",
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
    args = parser.parse_args(argv)

    if args.tools:
        print(json.dumps(openai_tool_definitions(), ensure_ascii=False, indent=2))
        return 0

    try:
        config = load_llm_config(
            Path(args.config),
            overrides={
                "api_key": args.api_key,
                "base_url": args.base_url,
                "model": args.model,
            },
        )
        if args.max_tool_rounds is not None:
            if not 1 <= args.max_tool_rounds <= 10:
                raise LLMConfigError("max_tool_rounds must be between 1 and 10")
            config = replace(config, max_tool_rounds=args.max_tool_rounds)
        if args.top_k is not None:
            if not 1 <= args.top_k <= 20:
                raise LLMConfigError("top_k must be between 1 and 20")
            config = replace(config, top_k=args.top_k)
    except (FileNotFoundError, LLMConfigError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    if args.check_config:
        print(json.dumps(config.redacted(), ensure_ascii=False, indent=2))
        if not config.has_api_key:
            print(
                "提示：api_key 为空。编辑 config/agent.json 或设置 "
                "PLOTSEEKER_LLM_API_KEY。",
                file=sys.stderr,
            )
        return 0

    if not config.has_api_key:
        print(
            "未配置 API Key。请编辑 config/agent.json 的 api_key，"
            "或设置环境变量 PLOTSEEKER_LLM_API_KEY。",
            file=sys.stderr,
        )
        return 2

    query = args.query_option or args.query
    if query is None or not query.strip():
        parser.error("query is required")

    embedder = None
    agent = None
    tools = None
    try:
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
        agent = LLMAgent(tools, config)
        result = agent.run(query)
    except LLMAPIError as exc:
        print(f"LLM 调用失败：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"Agent 运行失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if agent is not None:
            agent.close()
        if tools is not None:
            tools.close()
        if embedder is not None:
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
