"""Compare direct hybrid retrieval with the P9 agent on the P6 query set."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from evaluation.metrics import (
    first_relevant_rank,
    percentile,
    recall_at_k,
    reciprocal_rank,
    top1_accuracy,
)

REPORT_FORMAT = "p9-agent-evaluation-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _load_queries(path: Path) -> list[dict[str, Any]]:
    queries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in records if item["answerable"]]
    negative = [item for item in records if not item["answerable"]]
    latencies = [item["latency_ms"] for item in records]
    return {
        "query_count": len(records),
        "answerable_queries": len(answerable),
        "negative_queries": len(negative),
        "recall_at_5": round(
            sum(item["recall_at_5"] for item in answerable) / len(answerable), 4
        )
        if answerable
        else None,
        "mrr_at_10": round(
            sum(item["mrr_at_10"] for item in answerable) / len(answerable), 4
        )
        if answerable
        else None,
        "top1_accuracy": round(
            sum(item["top1"] for item in answerable) / len(answerable), 4
        )
        if answerable
        else None,
        "answerable_clarification_rate": round(
            sum(1 for item in answerable if item["clarified"]) / len(answerable), 4
        )
        if answerable
        else None,
        "negative_clarification_rate": round(
            sum(1 for item in negative if item["clarified"]) / len(negative), 4
        )
        if negative
        else None,
        "negative_false_positive_any": sum(
            1 for item in negative if item["top_ids"]
        ),
        "negative_false_positive_confident": sum(
            1
            for item in negative
            if item["top_ids"] and item["confidence"] in {"high", "medium"}
        ),
        "avg_tool_calls": round(
            sum(item["tool_call_count"] for item in records) / len(records), 2
        )
        if records
        else 0.0,
        "latency_ms": {
            "p50": round(percentile(latencies, 50) or 0.0, 2),
            "p95": round(percentile(latencies, 95) or 0.0, 2),
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        },
    }


def _rank_record(
    *,
    query: dict[str, Any],
    variant: str,
    candidates: Sequence[int],
    confidence: str,
    clarified: bool,
    tool_calls: int,
    steps: Sequence[str],
    latency_ms: float,
) -> dict[str, Any]:
    answerable = bool(query.get("answerable", True))
    acceptable = [int(value) for value in query.get("acceptable_subject_ids") or []]
    rank = first_relevant_rank(candidates, acceptable, 10) if answerable else None
    if not answerable:
        if clarified:
            failure = "correctly_clarified"
        elif candidates and confidence in {"high", "medium"}:
            failure = "confident_false_positive"
        elif candidates:
            failure = "weak_false_positive"
        else:
            failure = "no_candidates"
    elif clarified:
        failure = "clarified"
    elif not candidates:
        failure = "no_candidates"
    elif rank is None:
        failure = "not_in_top10"
    elif rank > 5:
        failure = "ranked_6_10"
    else:
        failure = None
    return {
        "query_id": query["query_id"],
        "query": query["query"],
        "clue_type": query["clue_type"],
        "answerable": answerable,
        "acceptable_subject_ids": acceptable,
        "variant": variant,
        "top_ids": list(candidates[:10]),
        "target_rank": rank,
        "recall_at_5": recall_at_k(rank, 5),
        "mrr_at_10": reciprocal_rank(rank, 10),
        "top1": top1_accuracy(rank),
        "confidence": confidence,
        "clarified": clarified,
        "tool_call_count": tool_calls,
        "steps": list(steps),
        "latency_ms": round(latency_ms, 2),
        "failure_reason": failure,
    }


def run_agent_evaluation(
    *,
    queries_path: Path,
    baseline_service,
    tools,
    agents: dict[str, Any],
    model_info: dict[str, Any],
    report_path: Path,
    markdown_path: Path,
    command: str = "",
    top_k: int = 10,
    candidate_k: int = 50,
) -> dict[str, Any]:
    queries = _load_queries(queries_path)
    details: dict[str, list[dict[str, Any]]] = {
        "baseline": [],
        **{name: [] for name in agents},
    }

    for query in queries:
        started = time.perf_counter()
        baseline = baseline_service.search(
            query["query"], mode="hybrid", top_k=top_k, candidate_k=candidate_k
        )
        details["baseline"].append(
            _rank_record(
                query=query,
                variant="baseline",
                candidates=[
                    int(item["subject_id"]) for item in baseline["candidates"]
                ],
                confidence=baseline["confidence"],
                clarified=False,
                tool_calls=0,
                steps=["hybrid"],
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        )
        for name, agent in agents.items():
            started = time.perf_counter()
            result = agent.run(query["query"], top_k=top_k)
            clarified = result.get("clarification") is not None
            candidates = [] if clarified else [
                int(item["subject_id"]) for item in result["candidates"]
            ]
            details[name].append(
                _rank_record(
                    query=query,
                    variant=name,
                    candidates=candidates,
                    confidence=result.get("confidence", "none"),
                    clarified=clarified,
                    tool_calls=len(result.get("tool_calls") or []),
                    steps=result.get("steps") or [],
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
            )

    summary = {name: _summary(records) for name, records in details.items()}
    conclusion = _conclusion(summary)
    report = {
        "format": REPORT_FORMAT,
        "generated_at": _now(),
        "command": command,
        "dataset": {
            "path": str(queries_path.resolve()),
            "query_count": len(queries),
        },
        "top_k": top_k,
        "candidate_k": candidate_k,
        "summary": summary,
        "details": details,
        "resources": {
            "model": model_info,
            "agent_versions": {
                name: getattr(agent, "agent_version", "unknown")
                for name, agent in agents.items()
            },
        },
        "conclusion": conclusion,
    }
    _atomic_write_json(report_path, report)
    report["_report_path"] = str(report_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(report) + "\n", encoding="utf-8")
    return report


def _conclusion(summary: dict[str, Any]) -> str:
    baseline = summary["baseline"]
    agent = summary.get("agent")
    if not agent:
        return "没有 agent 变体。"
    parts = []
    if (agent["recall_at_5"] or 0) > (baseline["recall_at_5"] or 0):
        parts.append("Agent Recall@5 高于直接 hybrid。")
    elif (agent["recall_at_5"] or 0) == (baseline["recall_at_5"] or 0):
        parts.append(
            "默认 Agent Recall@5 与直接 hybrid 持平，且平均工具调用 "
            f"{agent['avg_tool_calls']} 次，未证明默认启用有质量收益。"
        )
    else:
        parts.append("默认 Agent Recall@5 低于直接 hybrid，不建议默认启用。")
    if (agent["answerable_clarification_rate"] or 0) > 0:
        parts.append(
            f"可回答查询澄清率 {agent['answerable_clarification_rate']:.2%}。"
        )
    clarify = summary.get("agent_clarify_low")
    if clarify:
        parts.append(
            "保守澄清模式在开发集上保持 Recall@5，"
            f"可回答澄清率 {clarify['answerable_clarification_rate']:.2%}，"
            f"负样例澄清率 {clarify['negative_clarification_rate']:.2%}。"
            "该模式可作为低置信/不可回答查询的可选策略，"
            "但合成负样例不能替代真实用户验证。"
        )
    return " ".join(parts)


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# PlotSeeker P9 Agent 对比评测",
        "",
        "> 本报告由 `scripts/evaluate_agent.py` 自动生成，请勿手工编辑。",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 执行命令：`{report['command']}`",
        f"- 查询集：`{report['dataset']['path']}`，{report['dataset']['query_count']} 条",
        "- 基线：直接 hybrid 检索（无 Agent 编排）。",
        "- Agent 版本：确定性工具编排，不调用 LLM。",
        "",
        "## 1. 总体指标",
        "",
        "| 变体 | R@5 | MRR@10 | Top-1 | 可回答澄清率 | P50 | P95 | 平均工具调用 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, entry in summary.items():
        lines.append(
            f"| {name} | {entry['recall_at_5']} | {entry['mrr_at_10']} | "
            f"{entry['top1_accuracy']} | {entry['answerable_clarification_rate']} | "
            f"{entry['latency_ms']['p50']} ms | {entry['latency_ms']['p95']} ms | "
            f"{entry['avg_tool_calls']} |"
        )
    lines.extend(
        [
            "",
            "## 2. 不可回答查询",
            "",
            "| 变体 | 负样例澄清率 | Any FP | Confident FP |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, entry in summary.items():
        lines.append(
            f"| {name} | {entry['negative_clarification_rate']} | "
            f"{entry['negative_false_positive_any']} | "
            f"{entry['negative_false_positive_confident']} |"
        )
    lines.extend(
        [
            "",
            "## 3. 结论",
            "",
            report["conclusion"],
            "",
            "## 4. 失败样例",
            "",
        ]
    )
    for name, records in report["details"].items():
        failures = [item for item in records if item["failure_reason"]]
        lines.append(f"### {name}（{len(failures)}）")
        lines.append("")
        if not failures:
            lines.append("无。")
            lines.append("")
            continue
        lines.append("| query_id | 查询 | 类型 | 原因 | target_rank | 工具 |")
        lines.append("| --- | --- | --- | --- | ---: | --- |")
        for item in failures[:8]:
            query = item["query"].replace("|", "\\|")[:70]
            lines.append(
                f"| {item['query_id']} | {query} | {item['clue_type']} | "
                f"{item['failure_reason']} | "
                f"{item['target_rank'] if item['target_rank'] is not None else '-'} | "
                f"{' -> '.join(item['steps'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 5. 限制",
            "",
            "- 评测集由已索引文本生成，不是独立人工查询集。",
            "- Agent 不调用 LLM，工具调用成本按工具次数和额外延迟衡量，不含 token 成本。",
            "- 低置信候选默认仍返回；`agent_clarify_low` 才在低置信时直接追问。",
            "- 单进程测量，首次请求包含模型和索引加载时间。",
            "",
        ]
    )
    return "\n".join(lines)
