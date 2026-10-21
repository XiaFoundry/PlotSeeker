"""P6 baseline evaluation for keyword, vector and hybrid retrieval."""

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
from ingestion.resources import peak_rss_mb
from retrieval.keyword_index import KEYWORD_DB_PATH
from retrieval.search import KEYWORD_COVERAGE_THRESHOLD, MODES, RetrievalService

REPORT_FORMAT = "p6-evaluation-v1"


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
    queries: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            query_id = str(item.get("query_id") or "")
            if not query_id or query_id in seen:
                raise ValueError(f"query_id 缺失或重复：{path}:{line_number}")
            if not item.get("query"):
                raise ValueError(f"查询为空：{path}:{line_number}")
            item["answerable"] = bool(item.get("answerable", True))
            item["acceptable_subject_ids"] = [
                int(value) for value in item.get("acceptable_subject_ids") or []
            ]
            if item["answerable"] and not item["acceptable_subject_ids"]:
                raise ValueError(f"可回答查询缺少标签：{path}:{line_number}")
            queries.append(item)
            seen.add(query_id)
    return queries


def _failure_reason(
    *,
    answerable: bool,
    rank: int | None,
    candidates: Sequence[dict[str, Any]],
    confidence: str,
    mode: str,
) -> str | None:
    if answerable:
        if not candidates:
            return "no_candidates"
        if rank is None:
            return f"{mode}_not_in_top10"
        if rank > 5:
            return f"{mode}_ranked_6_10"
        return None
    if candidates:
        if confidence in {"high", "medium"}:
            return "confident_false_positive"
        return "weak_false_positive"
    return None


def _failure_category(item: dict[str, Any]) -> str | None:
    reason = item.get("failure_reason")
    if not reason:
        return None
    clue_type = str(item.get("clue_type") or "")
    if clue_type in {"name", "alias", "traditional"}:
        return "名称/别名/简繁"
    if clue_type.startswith("cross_type") or clue_type in {
        "character_summary",
        "episode_plot",
    }:
        return "关系扩展或跨类型定位"
    if reason.startswith("keyword"):
        return "关键词分词或召回"
    if reason.startswith("vector"):
        return "向量召回"
    if reason == "no_candidates":
        return "数据缺失或文本质量"
    if "ranked_6_10" in reason:
        return "排序"
    if "false_positive" in reason:
        return "问题歧义或负样例"
    return "其他"


def _aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in records if item["answerable"]]
    negative = [item for item in records if not item["answerable"]]
    latencies = [float(item["latency_ms"]) for item in records]
    clue_types = sorted({item["clue_type"] for item in answerable})
    by_clue: dict[str, Any] = {}
    for clue_type in clue_types:
        subset = [item for item in answerable if item["clue_type"] == clue_type]
        by_clue[clue_type] = _answerable_metrics(subset)
    result = _answerable_metrics(answerable)
    result.update(
        {
            "query_count": len(records),
            "answerable_queries": len(answerable),
            "negative_queries": len(negative),
            "negative_false_positive_any": sum(
                1 for item in negative if item["top_ids"]
            ),
            "negative_false_positive_confident": sum(
                1
                for item in negative
                if item["top_ids"]
                and item["confidence"] in {"high", "medium"}
            ),
            "negative_false_positive_rate_any": round(
                sum(1 for item in negative if item["top_ids"]) / len(negative), 4
            )
            if negative
            else None,
            "negative_false_positive_rate_confident": round(
                sum(
                    1
                    for item in negative
                    if item["top_ids"]
                    and item["confidence"] in {"high", "medium"}
                )
                / len(negative),
                4,
            )
            if negative
            else None,
            "latency_ms": {
                "p50": round(percentile(latencies, 50) or 0.0, 2),
                "p95": round(percentile(latencies, 95) or 0.0, 2),
                "mean": round(sum(latencies) / len(latencies), 2)
                if latencies
                else 0.0,
            },
            "by_clue_type": by_clue,
        }
    )
    return result


def _answerable_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "queries": 0,
            "recall_at_5": None,
            "mrr_at_10": None,
            "top1_accuracy": None,
        }
    return {
        "queries": len(records),
        "recall_at_5": round(
            sum(item["recall_at_5"] for item in records) / len(records), 4
        ),
        "mrr_at_10": round(
            sum(item["mrr_at_10"] for item in records) / len(records), 4
        ),
        "top1_accuracy": round(
            sum(item["top1"] for item in records) / len(records), 4
        ),
    }


def _evaluate_mode(
    service: RetrievalService,
    queries: Sequence[dict[str, Any]],
    mode: str,
    *,
    top_k: int,
    candidate_k: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for query in queries:
        started = time.perf_counter()
        result = service.search(
            query["query"], mode=mode, top_k=top_k, candidate_k=candidate_k
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        candidates = result["candidates"]
        retrieved = [int(item["subject_id"]) for item in candidates]
        acceptable = query["acceptable_subject_ids"]
        answerable = query["answerable"]
        rank = (
            first_relevant_rank(retrieved, acceptable, 10) if answerable else None
        )
        record = {
            "query_id": query["query_id"],
            "query": query["query"],
            "clue_type": query["clue_type"],
            "source_doc_id": query.get("source_doc_id"),
            "answerable": answerable,
            "acceptable_subject_ids": acceptable,
            "top_ids": retrieved[:10],
            "top_candidates": [
                {
                    "subject_id": item["subject_id"],
                    "name": item["name"],
                    "name_cn": item["name_cn"],
                    "score": item["score"],
                }
                for item in candidates[:3]
            ],
            "target_rank": rank,
            "recall_at_5": recall_at_k(rank, 5),
            "mrr_at_10": reciprocal_rank(rank, 10),
            "top1": top1_accuracy(rank),
            "confidence": result.get("confidence"),
            "keyword_coverage": result.get("keyword_coverage"),
            "vector_top_similarity": result.get("vector_top_similarity"),
            "warnings": result.get("warnings") or [],
            "latency_ms": round(latency_ms, 2),
            "failure_reason": _failure_reason(
                answerable=answerable,
                rank=rank,
                candidates=candidates,
                confidence=result.get("confidence", "none"),
                mode=mode,
            ),
        }
        record["failure_category"] = _failure_category(record)
        records.append(record)
    return {
        "summary": _aggregate(records),
        "records": records,
    }


def _relative_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, dirs, files in path.walk():
        if "_tmp" in root.parts:
            continue
        for filename in files:
            try:
                total += (root / filename).stat().st_size
            except OSError:
                pass
    return total


def _format_metrics_table(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        "| 模式 | 查询数 | Recall@5 | MRR@10 | Top-1 | P50 (ms) | P95 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in rows:
        rec = summary["recall_at_5"]
        mrr = summary["mrr_at_10"]
        top1 = summary["top1_accuracy"]
        latency = summary["latency_ms"]
        lines.append(
            f"| {name} | {summary['answerable_queries']} | "
            f"{rec if rec is not None else '-'} | "
            f"{mrr if mrr is not None else '-'} | "
            f"{top1 if top1 is not None else '-'} | "
            f"{latency['p50']} | {latency['p95']} |"
        )
    return "\n".join(lines)


def _format_clue_table(summary: dict[str, Any], modes: Sequence[str]) -> str:
    clue_types = sorted(
        {
            clue_type
            for mode in modes
            for clue_type in summary[mode]["by_clue_type"]
        }
    )
    header = ["线索类型", "查询数"]
    for mode in modes:
        header.extend([f"{mode} R@5", f"{mode} MRR", f"{mode} Top1"])
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
    ]
    for clue_type in clue_types:
        row = [clue_type, ""]
        count = 0
        for mode in modes:
            entry = summary[mode]["by_clue_type"].get(clue_type, {})
            count = max(count, int(entry.get("queries") or 0))
            row.extend(
                [
                    entry.get("recall_at_5"),
                    entry.get("mrr_at_10"),
                    entry.get("top1_accuracy"),
                ]
            )
        row[1] = str(count)
        lines.append("| " + " | ".join("-" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def _format_failures(
    details: dict[str, list[dict[str, Any]]], modes: Sequence[str], limit: int = 8
) -> str:
    lines: list[str] = []
    for mode in modes:
        failures = [item for item in details[mode] if item["failure_reason"]]
        lines.append(f"### {mode} 失败样例（{len(failures)} 条）")
        if not failures:
            lines.append("无。")
            continue
        lines.append(
            "| query_id | 查询 | 线索类型 | 失败原因 | 归因 | target_rank | top_ids |"
        )
        lines.append("| --- | --- | --- | --- | --- | ---: | --- |")
        for item in failures[:limit]:
            query = item["query"].replace("|", "\\|")[:80]
            lines.append(
                f"| {item['query_id']} | {query} | {item['clue_type']} | "
                f"{item['failure_reason']} | {item.get('failure_category') or '-'} | "
                f"{item['target_rank'] if item['target_rank'] is not None else '-'} | "
                f"{', '.join(str(value) for value in item['top_ids'][:3])} |"
            )
        lines.append("")
    return "\n".join(lines)


def _comparison_note(summary: dict[str, Any]) -> str:
    keyword = summary["keyword"]["recall_at_5"]
    vector = summary["vector"]["recall_at_5"]
    hybrid = summary["hybrid"]["recall_at_5"]
    if keyword is None or vector is None or hybrid is None:
        return "至少一个模式没有可回答查询结果，无法比较。"
    best_single = max(keyword, vector)
    keyword_summary = summary["keyword"]
    vector_summary = summary["vector"]
    hybrid_summary = summary["hybrid"]
    best_single = max(keyword, vector)
    best_mrr = max(keyword_summary["mrr_at_10"], vector_summary["mrr_at_10"])
    best_top1 = max(
        keyword_summary["top1_accuracy"], vector_summary["top1_accuracy"]
    )
    if hybrid > best_single:
        return (
            f"混合检索 Recall@5={hybrid} 高于最佳单路 Recall@5={best_single}，"
            "本次实测存在提升。"
        )
    if hybrid == best_single:
        extra = ""
        if hybrid_summary["mrr_at_10"] > best_mrr or hybrid_summary["top1_accuracy"] > best_top1:
            extra = " MRR 或 Top-1 有提升，但 Recall@5 未提升。"
        else:
            extra = " 本次实测未观察到提升。"
        return (
            f"混合检索 Recall@5={hybrid} 与最佳单路持平。" + extra
        )
    return (
        f"混合检索 Recall@5={hybrid} 低于最佳单路 Recall@5={best_single}，"
        "本次实测出现退化，需要检查融合权重和候选规模。"
    )


def _render_markdown(
    report: dict[str, Any],
    summary: dict[str, Any],
    details: dict[str, list[dict[str, Any]]],
    failure_path: Path,
    modes: Sequence[str],
) -> str:
    generated = report["generated_at"]
    dataset = report["dataset"]
    resources = report["resources"]
    lines = [
        "# PlotSeeker P6 检索评测报告",
        "",
        "> 本报告由 `scripts/evaluate.py` 自动生成，请勿手工编辑。",
        "",
        f"- 生成时间：{generated}",
        f"- 执行命令：`{report['command']}`",
        f"- 查询集：`{dataset['path']}`，{dataset['query_count']} 条"
        f"（可回答 {dataset['answerable']}，不可回答 {dataset['unanswerable']}）",
        f"- 索引范围：P4 10,000 篇抽样文档 + 同批 FTS5 关键词索引",
        f"- P4 config_id：`{report.get('p4_config_id')}`",
        "",
        "## 1. 总体指标",
        "",
        _format_metrics_table([(mode, summary[mode]) for mode in modes]),
        "",
        f"**结论：** {_comparison_note(summary)}",
        "",
        "## 2. 按线索类型",
        "",
        _format_clue_table(summary, modes),
        "",
        "## 3. 不可回答查询",
        "",
        "| 模式 | Any false positive | Confident false positive | Any FP rate | Confident FP rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in modes:
        entry = summary[mode]
        lines.append(
            f"| {mode} | {entry['negative_false_positive_any']} | "
            f"{entry['negative_false_positive_confident']} | "
            f"{entry['negative_false_positive_rate_any']} | "
            f"{entry['negative_false_positive_rate_confident']} |"
        )
    lines.extend(
        [
            "",
            "## 4. 资源与延迟",
            "",
            "| 项目 | 数值 |",
            "| --- | ---: |",
            f"| 关键词索引体积 | {resources['keyword_db_size_bytes'] / 1024 / 1024:.1f} MiB |",
            f"| P4 向量索引体积 | {resources['vector_store_size_bytes'] / 1024 / 1024:.1f} MiB |",
            f"| 进程峰值内存 | {resources['peak_rss_mb']} MiB |",
            f"| 模型 | {resources['model'].get('model_id')} / {resources['model'].get('revision')} |",
            f"| 设备 | {resources['model'].get('device')} |",
            "",
            "## 5. 失败案例",
            "",
            f"完整失败记录：`{failure_path}`",
            "",
            _format_failures(details, modes),
            "## 6. 评测限制",
            "",
            "- 查询由已索引文档文本生成，不是独立人工查询集；部分查询与索引文本共享措辞。",
            "- 仅覆盖 P4 的 10,000 篇抽样文档，不代表 84.6 万篇全量数据。",
            "- 可接受答案只使用 P3 文档记录的主作品，角色跨作品关系未全部列入可接受集合。",
            "- 不可回答查询是合成负样例，可能偶然命中真实词汇。",
            "- 置信度使用关键词覆盖率阈值；阈值在开发集上校准，不是独立验证。",
            "- Any false positive 包含低置信度近邻；Confident false positive 才反映高置信误报。",
            "- 延迟为预热后单进程测量，首次请求还包含模型和索引加载时间。",
            "- 本报告是 P6 开发基线，不是最终检索质量结论。",
            "",
            "## 7. 数据文件",
            "",
            f"- JSON 报告：`{report.get('_report_path', '')}`",
            f"- 失败记录：`{failure_path}`",
            "",
        ]
    )
    return "\n".join(lines)


def _render_failures(
    details: dict[str, list[dict[str, Any]]], modes: Sequence[str]
) -> str:
    lines = ["# P6 失败案例记录", ""]
    for mode in modes:
        failures = [item for item in details[mode] if item["failure_reason"]]
        lines.append(f"## {mode}（{len(failures)}）")
        lines.append("")
        if not failures:
            lines.append("无。")
            lines.append("")
            continue
        for item in failures:
            lines.append(f"### {item['query_id']} {item['clue_type']}")
            lines.append("")
            lines.append(f"- 查询：{item['query']}")
            lines.append(f"- 失败原因：`{item['failure_reason']}`")
            lines.append(f"- target_rank：{item['target_rank']}")
            lines.append(f"- top_ids：{item['top_ids']}")
            lines.append(f"- confidence：{item['confidence']}")
            lines.append("")
    return "\n".join(lines)


def run_evaluation(
    *,
    queries_path: Path,
    service: RetrievalService,
    model_info: dict[str, Any],
    command: str = "",
    modes: Sequence[str] = MODES,
    top_k: int = 10,
    candidate_k: int = 50,
    report_path: Path,
    markdown_path: Path,
    failures_path: Path,
    keyword_db_path: Path = KEYWORD_DB_PATH,
) -> dict[str, Any]:
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
    queries = _load_queries(queries_path)
    started = time.perf_counter()

    for mode in modes:
        try:
            service.search("时间循环", mode=mode, top_k=1, candidate_k=5)
        except Exception:
            pass

    details: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {}
    for mode in modes:
        mode_result = _evaluate_mode(
            service, queries, mode, top_k=top_k, candidate_k=candidate_k
        )
        details[mode] = mode_result["records"]
        summary[mode] = mode_result["summary"]

    elapsed = time.perf_counter() - started
    p4_config_id = None
    try:
        checkpoint_path = (
            service.checkpoint_path
            if service.checkpoint_path is not None
            else service.store_dir / "p4_checkpoint.json"
        )
        p4_config_id = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        ).get("config_id")
    except Exception:
        p4_config_id = None
    report = {
        "format": REPORT_FORMAT,
        "generated_at": _now(),
        "command": command,
        "elapsed_sec": round(elapsed, 2),
        "p4_config_id": p4_config_id,
        "dataset": {
            "path": str(queries_path.resolve()),
            "query_count": len(queries),
            "answerable": sum(1 for item in queries if item["answerable"]),
            "unanswerable": sum(1 for item in queries if not item["answerable"]),
        },
        "modes": list(modes),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "summary": summary,
        "details": details,
        "resources": {
            "keyword_db_size_bytes": _relative_size(keyword_db_path),
            "vector_store_size_bytes": _relative_size(service.store_dir),
            "peak_rss_mb": round(peak_rss_mb() or 0.0, 1),
            "keyword_coverage_threshold": KEYWORD_COVERAGE_THRESHOLD,
            "model": model_info,
        },
    }
    _atomic_write_json(report_path, report)
    report["_report_path"] = str(report_path)

    markdown = _render_markdown(
        report, summary, details, failures_path, modes
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.write_text(_render_failures(details, modes) + "\n", encoding="utf-8")
    return report
