"""Deterministic single-agent policy over P5/P9 retrieval tools.

There is no LLM in this version.  The agent routes between subject, character
and episode retrieval, falls back when confidence is low, optionally fetches
related subjects, and asks for clarification when evidence is insufficient.
"""

from __future__ import annotations

import re
import time
from typing import Any

from retrieval.preprocess import normalize_text

CHARACTER_RE = re.compile(
    r"(角色|人物|主角|女主|男主|声优|能力|性格|特征|记得有个)"
)
EPISODE_RE = re.compile(
    r"(第.{1,4}话|一话|某一话|章节|集数|哪一集|记得有一话|剧情片段)"
)
RELATION_RE = re.compile(r"(相关|类似|续作|同系列|关系|前作|续集)")

AGENT_VERSION = "p9-v1"


class PlotSeekerAgent:
    def __init__(
        self,
        tools,
        *,
        max_steps: int = 4,
        top_k: int = 5,
        candidate_k: int = 30,
        clarify_on_low: bool = False,
    ) -> None:
        self.tools = tools
        self.agent_version = AGENT_VERSION
        self.clarify_on_low = clarify_on_low
        self.max_steps = max(1, int(max_steps))
        self.top_k = max(1, int(top_k))
        self.candidate_k = max(1, int(candidate_k))

    def _route(self, query: str) -> str:
        if EPISODE_RE.search(query):
            return "episodes"
        if CHARACTER_RE.search(query):
            return "characters"
        return "subjects"

    def _record(
        self,
        tool_calls: list[dict[str, Any]],
        tool: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        latency_ms: float,
    ) -> None:
        if tool.startswith("search_"):
            summary = [
                item.get("doc_id") or item.get("subject_id")
                for item in (result.get("results") or result.get("candidates") or [])[:3]
            ]
        elif tool == "get_related_subjects":
            summary = [
                item.get("subject_id") for item in result.get("results", [])[:3]
            ]
        else:
            summary = []
        tool_calls.append(
            {
                "tool": tool,
                "arguments": arguments,
                "count": int(result.get("count", len(summary))),
                "confidence": result.get("confidence"),
                "latency_ms": round(latency_ms, 2),
                "summary": summary,
            }
        )

    def _call(
        self,
        tool: str,
        query: str,
        tool_calls: list[dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if tool == "subjects":
            result = self.tools.search_subjects(
                query, top_k=top_k or self.candidate_k
            )
            tool_name = "search_subjects"
        elif tool == "characters":
            result = self.tools.search_characters(
                query, top_k=top_k or self.candidate_k
            )
            tool_name = "search_characters"
        elif tool == "episodes":
            result = self.tools.search_episodes(
                query, top_k=top_k or self.candidate_k
            )
            tool_name = "search_episodes"
        else:
            raise ValueError(f"unknown tool route: {tool}")
        self._record(
            tool_calls,
            tool_name,
            {"query": query, "top_k": top_k or self.candidate_k},
            result,
            (time.perf_counter() - started) * 1000.0,
        )
        return result

    def _entity_candidates(
        self, tool_result: dict[str, Any], source_tool: str
    ) -> list[dict[str, Any]]:
        grouped: dict[int, dict[str, Any]] = {}
        for item in tool_result.get("results") or []:
            subject_id = int(item["subject_id"])
            candidate = grouped.setdefault(
                subject_id,
                {
                    "subject_id": subject_id,
                    "name": item.get("subject_name") or "",
                    "name_cn": item.get("subject_name_cn") or "",
                    "score": 0.0,
                    "keyword_rank": None,
                    "vector_rank": None,
                    "evidence": [],
                },
            )
            candidate["score"] = max(candidate["score"], float(item["score"]))
            keyword_rank = item.get("keyword_rank")
            vector_rank = item.get("vector_rank")
            if keyword_rank is not None:
                current = candidate["keyword_rank"]
                candidate["keyword_rank"] = (
                    keyword_rank if current is None else min(current, keyword_rank)
                )
            if vector_rank is not None:
                current = candidate["vector_rank"]
                candidate["vector_rank"] = (
                    vector_rank if current is None else min(current, vector_rank)
                )
            if len(candidate["evidence"]) < 4:
                candidate["evidence"].append(
                    {
                        "doc_id": item["doc_id"],
                        "entity_type": item["entity_type"],
                        "entity_id": item["entity_id"],
                        "subject_id": subject_id,
                        "source": source_tool,
                        "rank": min(
                            value
                            for value in (keyword_rank, vector_rank)
                            if value is not None
                        ),
                        "score": float(item["score"]),
                        "snippet": item.get("snippet") or "",
                        "relation_path": item.get("relation_path") or "",
                        "contribution": float(item["score"]),
                    }
                )
        candidates = list(grouped.values())
        candidates.sort(key=lambda item: (-item["score"], item["subject_id"]))
        return candidates

    def _subject_candidates(self, tool_result: dict[str, Any]) -> list[dict[str, Any]]:
        return list(tool_result.get("candidates") or [])

    def _merge_candidates(
        self, candidate_lists: list[list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        merged: dict[int, dict[str, Any]] = {}
        for candidates in candidate_lists:
            for candidate in candidates:
                subject_id = int(candidate["subject_id"])
                existing = merged.get(subject_id)
                if existing is None:
                    merged[subject_id] = {
                        **candidate,
                        "evidence": list(candidate.get("evidence") or []),
                    }
                    continue
                existing["score"] = max(
                    float(existing["score"]), float(candidate["score"])
                )
                for key in ("keyword_rank", "vector_rank"):
                    value = candidate.get(key)
                    if value is not None:
                        current = existing.get(key)
                        existing[key] = value if current is None else min(current, value)
                existing_evidence = existing.get("evidence") or []
                seen = {item.get("doc_id") for item in existing_evidence}
                for item in candidate.get("evidence") or []:
                    if item.get("doc_id") not in seen and len(existing_evidence) < 6:
                        existing_evidence.append(item)
                existing["evidence"] = existing_evidence
        result = list(merged.values())
        result.sort(key=lambda item: (-item["score"], item["subject_id"]))
        return result

    def _confidence(self, tool_result: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
        if not candidates:
            return "none"
        if tool_result.get("confidence"):
            return str(tool_result["confidence"])
        if any(item.get("keyword_rank") and item.get("vector_rank") for item in candidates):
            return "high"
        if any(item.get("keyword_rank") for item in candidates):
            return "medium"
        return "low"

    def _clarify(
        self,
        query: str,
        reason: str,
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        question = (
            "没有找到足够可追溯的候选作品。请补充角色名、章节剧情、"
            "作品类型或大致年份。"
            if reason == "no_candidates"
            else "当前有多个可能作品，能否补充角色名、章节细节或作品类型？"
        )
        started = time.perf_counter()
        result = self.tools.ask_user_for_clarification(
            question,
            options=["补充角色名", "补充章节剧情", "补充作品类型/年份"],
            reason=reason,
        )
        self._record(
            tool_calls,
            "ask_user_for_clarification",
            {"query": query, "reason": reason},
            result,
            (time.perf_counter() - started) * 1000.0,
        )
        return result

    def _answer_text(
        self,
        candidates: list[dict[str, Any]],
        related_subjects: list[dict[str, Any]],
    ) -> str:
        if not candidates:
            return ""
        top = candidates[0]
        name = top.get("name_cn") or top.get("name") or f"subject:{top['subject_id']}"
        lines = [f"最可能是《{name}》（subject:{top['subject_id']}）。"]
        evidence = top.get("evidence") or []
        if evidence:
            lines.append("依据：")
            for item in evidence[:2]:
                snippet = (item.get("snippet") or "").replace("\n", " ")[:100]
                lines.append(
                    f"- [{item.get('source')}] {item.get('doc_id')} "
                    f"{item.get('relation_path')} {snippet}"
                )
        if related_subjects:
            related = "、".join(
                item.get("name_cn") or item.get("name") or str(item["subject_id"])
                for item in related_subjects[:5]
            )
            lines.append(f"相关作品：{related}")
        return "\n".join(lines)

    def run(self, query: str, *, top_k: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        tool_calls: list[dict[str, Any]] = []
        steps: list[str] = []
        normalized = normalize_text(query)
        limit = top_k or self.top_k
        if not normalized:
            clarification = self._clarify(query, "no_candidates", tool_calls)
            return {
                "query": query,
                "agent_version": AGENT_VERSION,
                "answer": clarification["question"],
                "confidence": "none",
                "warnings": [],
                "candidates": [],
                "clarification": clarification,
                "related_subjects": [],
                "tool_calls": tool_calls,
                "steps": steps,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "used_llm": False,
            }

        primary = self._route(query)
        first = self._call(primary, query, tool_calls)
        steps.append(primary)
        if primary == "subjects":
            candidates = self._subject_candidates(first)
        else:
            candidates = self._entity_candidates(first, f"search_{primary}")
        confidence = self._confidence(first, candidates)

        if (not candidates or confidence == "low") and len(steps) < self.max_steps:
            alternatives = [
                item
                for item in ("characters", "episodes", "subjects")
                if item != primary
            ]
            for alternative in alternatives:
                if len(steps) >= self.max_steps:
                    break
                fallback = self._call(alternative, query, tool_calls)
                steps.append(alternative)
                if alternative == "subjects":
                    fallback_candidates = self._subject_candidates(fallback)
                else:
                    fallback_candidates = self._entity_candidates(
                        fallback, f"search_{alternative}"
                    )
                candidates = self._merge_candidates(
                    [candidates, fallback_candidates]
                )
                fallback_confidence = self._confidence(
                    fallback, fallback_candidates
                )
                if fallback_confidence in {"high", "medium"}:
                    confidence = fallback_confidence
                elif confidence == "none" and fallback_candidates:
                    confidence = fallback_confidence
                if candidates and confidence in {"high", "medium"}:
                    break

        related_subjects: list[dict[str, Any]] = []
        if RELATION_RE.search(query) and candidates:
            top_subject_id = int(candidates[0]["subject_id"])
            started_relation = time.perf_counter()
            related = self.tools.get_related_subjects(top_subject_id, limit=10)
            self._record(
                tool_calls,
                "get_related_subjects",
                {"subject_id": top_subject_id, "limit": 10},
                related,
                (time.perf_counter() - started_relation) * 1000.0,
            )
            related_subjects = related.get("results") or []

        candidates.sort(key=lambda item: (-item["score"], item["subject_id"]))
        candidates = candidates[:limit]

        clarification = None
        warnings: list[str] = []
        if not candidates:
            clarification = self._clarify(query, "no_candidates", tool_calls)
            confidence = "none"
        elif confidence == "low" and self.clarify_on_low:
            clarification = self._clarify(query, "ambiguous", tool_calls)
        elif confidence == "low":
            warnings.append("low_confidence_candidates")

        answer = (
            clarification["question"]
            if clarification is not None
            else self._answer_text(candidates, related_subjects)
        )
        return {
            "query": query,
            "agent_version": AGENT_VERSION,
            "answer": answer,
            "confidence": confidence,
            "warnings": warnings,
            "candidates": candidates,
            "clarification": clarification,
            "related_subjects": related_subjects,
            "tool_calls": tool_calls,
            "steps": steps,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "used_llm": False,
        }
