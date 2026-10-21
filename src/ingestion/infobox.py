"""Tolerant Bangumi infobox parsing shared by audit and document building."""

from __future__ import annotations

import re
from typing import Any

INFOBOX_KEY_RE = re.compile(r"^\s*\|\s*([^=|\n]+?)\s*=(.*)$")
INFOBOX_START_RE = re.compile(r"^\s*\{\{\s*Infobox", re.IGNORECASE)

NAME_KEYS = (
    "简体中文名",
    "中文名",
    "第二中文名",
    "繁体中文名",
    "繁體中文名",
    "别名",
    "別名",
    "日文名",
    "英文名",
    "英文名二",
    "罗马字",
    "羅馬字",
    "纯假名",
    "純假名",
    "昵称",
    "其他名义",
    "其他名義",
    "原名",
)

CN_NAME_KEYS = ("简体中文名", "中文名", "第二中文名", "繁体中文名", "繁體中文名")


def _clean_infobox_value(value: str) -> str:
    text = value.strip()
    while text.endswith("}"):
        text = text[:-1].rstrip()
    return text


def parse_infobox(text: Any) -> dict[str, str]:
    """Tolerantly parse a Bangumi infobox template into ``key -> value``.

    Handles the common multi-line form, trailing template braces, and a
    single-line fallback. Malformed input returns whatever key/value pairs
    could be recovered; it never raises.
    """

    if not isinstance(text, str) or not text.strip():
        return {}

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    result: dict[str, str] = {}
    current: str | None = None

    for line in normalized.split("\n"):
        stripped = line.strip()
        if current is None and INFOBOX_START_RE.match(line):
            continue
        match = INFOBOX_KEY_RE.match(line)
        if match:
            key = match.group(1).strip()
            current = key
            result[key] = _clean_infobox_value(match.group(2))
            continue
        if current is None:
            continue
        if stripped.startswith("}}"):
            continue
        result[current] = (result[current] + "\n" + line).strip()

    if not result:
        body = normalized.strip().strip("{}").strip()
        body = re.sub(r"^\s*Infobox[^\n|]*", "", body, flags=re.IGNORECASE)
        for part in re.split(r"\|(?=\s*[^|=]+?=)", body):
            key, sep, value = part.partition("=")
            if sep and key.strip():
                result[key.strip()] = _clean_infobox_value(value)

    return {
        key: _clean_infobox_value(value)
        for key, value in result.items()
        if key and value.strip()
    }


def parse_aliases(value: Any) -> list[str]:
    """Extract alias values from an infobox ``别名`` field.

    Bangumi stores aliases either as bracketed multi-line entries such as
    ``[英文名|Lelouch]`` or, less commonly, as plain newline/comma separated
    text. Label-only entries (``[纯假名|]``) are skipped.
    """

    if not isinstance(value, str) or not value.strip():
        return []

    entries = re.findall(r"\[([^\[\]]*)\]", value, flags=re.DOTALL)
    aliases: list[str] = []
    if entries:
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            if "|" in entry:
                _, _, raw_value = entry.partition("|")
                raw_value = raw_value.strip()
                if raw_value:
                    aliases.append(raw_value)
            else:
                aliases.append(entry)
    else:
        body = value.strip().strip("{}").strip()
        for part in re.split(r"[\n,，、;；]+", body):
            part = part.strip().strip("[]")
            if not part:
                continue
            if "=" in part:
                _, _, part = part.partition("=")
                part = part.strip()
            if part:
                aliases.append(part)

    seen: set[str] = set()
    deduped: list[str] = []
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            deduped.append(alias)
    return deduped


def cn_name_from_infobox(info: dict[str, str]) -> str:
    for key in CN_NAME_KEYS:
        value = info.get(key, "").strip()
        if value:
            return parse_aliases(value)[0] if value.startswith("[") else value
    return ""
