"""Heuristics for separating episode plot text from staff/credit text."""

from __future__ import annotations

import re
from typing import Any

CREDIT_MARKER_RE = re.compile(
    r"(シナリオ|脚本|コンテ|絵コンテ|シリーズ構成|演出|作画監督|総作画監督|作監|"
    r"キャラクターデザイン|メカ作監|美術監督|撮影監督|音響監督|色彩設計|編集|"
    r"音楽|主題歌|エンディング|オープニング|挿入歌|声の出演|キャスト|スタッフ|"
    r"制作|プロデューサー|プロダクション|監督|アニメーション制作|制作協力|"
    r"原作|企画|原画|動画|背景|仕上|特殊効果|CG|撮影|音響|効果|調整|録音|"
    r"脚本協力|演出助手|設定|メカデザイン|衣装デザイン|アイキャッチ|美術|現像|"
    r"オンライン編集|オフライン編集|MA|音響制作|音楽制作|製作)"
)

QUALITY_LABELS = {
    "empty": "空描述",
    "credit_only": "仅制作/人员信息",
    "mixed": "制作信息 + 剧情文本",
    "narrative": "疑似剧情",
    "other": "过短或无法判定",
}

NARRATIVE_PUNCT_RE = re.compile(r"[。！？!?]")


def _line_is_narrative(line: str) -> bool:
    return bool(NARRATIVE_PUNCT_RE.search(line))


def _line_is_credit(line: str) -> bool:
    return len(line) <= 200 and bool(CREDIT_MARKER_RE.search(line)) and not _line_is_narrative(line)


def classify_episode_description(description: Any) -> str:
    """Heuristically separate plot text from staff/credit text.

    A line that contains sentence-ending punctuation is treated as narrative
    even when it also mentions staff labels, because many Chinese/Japanese
    summaries append ``脚本：...`` inline to the plot text.
    """

    if not isinstance(description, str) or not description.strip():
        return "empty"

    credit_lines = 0
    narrative_lines = 0
    for line in description.splitlines():
        line = line.strip()
        if not line:
            continue
        if _line_is_credit(line):
            credit_lines += 1
        elif len(line) >= 12:
            narrative_lines += 1

    if credit_lines and narrative_lines:
        return "mixed"
    if credit_lines:
        return "credit_only"
    if narrative_lines:
        return "narrative"
    return "other"


def clean_episode_description(description: Any) -> tuple[str, int]:
    """Remove credit-only lines and return ``(cleaned_text, removed_lines)``."""

    if not isinstance(description, str) or not description.strip():
        return "", 0

    kept: list[str] = []
    removed = 0
    for line in description.splitlines():
        line = line.strip()
        if not line:
            continue
        if _line_is_credit(line):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed
