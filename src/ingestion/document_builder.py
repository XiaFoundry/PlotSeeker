"""Build retrieval documents from the structured SQLite database.

P3 produces three JSONL collections:

* ``subjects.jsonl``: subjects with a non-empty summary.
* ``characters.jsonl``: characters with a non-empty summary and at least one
  resolvable subject relation. One character produces one document that lists
  its associated subjects.
* ``episodes.jsonl``: episodes with a valid subject and narrative description.
  Credit-only or non-narrative descriptions are filtered out and the filter
  reason is counted in the manifest.

Each document keeps the original entity ID, source file, a stable ``doc_id``,
and enough metadata for later vector indexing and evidence assembly. Document
text is capped; every truncation is recorded in document metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import sys
import time
from array import array
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from config import DOCUMENTS_DIR, SQLITE_PATH
from ingestion.infobox import cn_name_from_infobox, parse_aliases, parse_infobox
from ingestion.resources import peak_rss_mb
from ingestion.text_quality import classify_episode_description, clean_episode_description

RULES_VERSION = "p3-v1"
SCHEMA_VERSION = 1

SUBJECT_SUMMARY_MAX_CHARS = 4000
CHARACTER_SUMMARY_MAX_CHARS = 3000
EPISODE_DESCRIPTION_MAX_CHARS = 2000
TOTAL_TEXT_MAX_CHARS = {
    "subject": 6000,
    "character": 5000,
    "episode": 4000,
}
MAX_ALIASES = 30
MAX_ALIAS_CHARS = 200
MAX_TAGS = 50
MAX_TAG_CHARS = 100
MAX_SUBJECTS_IN_TEXT = 10
MAX_SUBJECTS_IN_METADATA = 50
MIN_EPISODE_CLEANED_CHARS = 12
EPISODE_INCLUDED_QUALITY = {"narrative", "mixed"}

SAMPLE_SIZE_BY_TYPE = {"subject": 20, "character": 15, "episode": 15}

MAIN_FILES = {
    "subject": "subject.jsonlines",
    "character": "character.jsonlines",
    "episode": "episode.jsonlines",
}

SUBJECT_SQL = """
SELECT id, name, name_cn, infobox_raw, summary, tags_json,
       type, platform, nsfw, score, rank, date
FROM subject
ORDER BY id
"""

CHARACTER_SQL = """
SELECT c.id, c.name, c.infobox_raw, c.summary,
       s.id AS subject_id, s.name AS subject_name, s.name_cn AS subject_name_cn
FROM character c
LEFT JOIN subject_characters sc ON sc.character_id = c.id
LEFT JOIN subject s ON s.id = sc.subject_id
ORDER BY c.id, s.id
"""

EPISODE_SQL = """
SELECT e.id, e.name, e.name_cn, e.description, e.airdate, e.sort,
       e.subject_id, s.id AS resolved_subject_id, s.name, s.name_cn
FROM episode e
LEFT JOIN subject s ON s.id = e.subject_id
ORDER BY e.id
"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


class _TextStats:
    __slots__ = ("total", "total_chars", "_values")

    def __init__(self) -> None:
        self.total = 0
        self.total_chars = 0
        self._values = array("I")

    def add(self, length: int) -> None:
        self.total += 1
        self.total_chars += length
        self._values.append(min(max(length, 0), 4_000_000_000))

    def summary(self) -> dict[str, Any]:
        if not self._values:
            return {
                "count": 0,
                "min": 0,
                "p50": 0,
                "p90": 0,
                "p99": 0,
                "max": 0,
                "mean": 0.0,
            }
        values = sorted(self._values)
        return {
            "count": len(values),
            "min": values[0],
            "p50": _percentile(values, 50),
            "p90": _percentile(values, 90),
            "p99": _percentile(values, 99),
            "max": values[-1],
            "mean": round(self.total_chars / len(values), 1),
        }


def _percentile(sorted_values: list[int], p: float) -> int:
    if not sorted_values:
        return 0
    index = int(round((len(sorted_values) - 1) * p / 100.0))
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


class _Reservoir:
    def __init__(self, k: int, seed: int) -> None:
        self.k = max(k, 0)
        self._rng = random.Random(seed)
        self.seen = 0
        self.items: list[Any] = []

    def add(self, item: Any) -> None:
        self.seen += 1
        if self.k == 0:
            return
        if len(self.items) < self.k:
            self.items.append(item)
            return
        index = self._rng.randrange(self.seen)
        if index < self.k:
            self.items[index] = item


def _clean_inline(value: Any, max_chars: int | None = None) -> tuple[str, bool]:
    if not isinstance(value, str):
        return "", False
    text = " ".join(value.split())
    if not text:
        return "", False
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars].rstrip() + "…", True
    return text, False


def _strip_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _truncate_text(value: Any, max_chars: int) -> tuple[str, bool]:
    if not isinstance(value, str):
        return "", False
    text = value.strip()
    if not text:
        return "", False
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "…", True
    return text, False


def _json_list(value: Any) -> list[Any]:
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _tag_names(tags_json: Any) -> tuple[list[str], bool]:
    names: list[str] = []
    seen: set[str] = set()
    truncated = False
    for item in _json_list(tags_json):
        if isinstance(item, dict):
            raw_name = item.get("name")
        else:
            raw_name = item
        name, was_truncated = _clean_inline(raw_name, MAX_TAG_CHARS)
        truncated = truncated or was_truncated
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= MAX_TAGS:
            truncated = True
            break
    return names, truncated


def _infobox_names(infobox_raw: Any) -> list[str]:
    info = parse_infobox(infobox_raw)
    if not info:
        return []
    values: list[str] = []
    alias_value = info.get("别名") or info.get("別名") or ""
    values.extend(parse_aliases(alias_value))
    for key in (
        "简体中文名",
        "中文名",
        "第二中文名",
        "繁体中文名",
        "繁體中文名",
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
    ):
        value = info.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        values.extend(parse_aliases(value) if "[" in value else [value])
    return values


def _prepare_aliases(
    infobox_raw: Any,
    *,
    exclude: tuple[str, ...],
) -> tuple[list[str], bool, int]:
    excluded = {value.strip() for value in exclude if value and value.strip()}
    all_aliases: list[str] = []
    seen: set[str] = set()
    per_alias_truncated = False
    for value in _infobox_names(infobox_raw):
        alias, was_truncated = _clean_inline(value, MAX_ALIAS_CHARS)
        per_alias_truncated = per_alias_truncated or was_truncated
        if not alias or alias in excluded or alias in seen:
            continue
        seen.add(alias)
        all_aliases.append(alias)
    total = len(all_aliases)
    truncated = per_alias_truncated or total > MAX_ALIASES
    return all_aliases[:MAX_ALIASES], truncated, total


def _compose_text(parts: list[str], max_chars: int) -> tuple[str, bool]:
    text = "\n".join(part for part in parts if part)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "…", True
    return text, False


def _doc_id(entity_type: str, entity_id: int) -> str:
    return f"{entity_type}:{entity_id}"


def _mark_truncation(fields: set[str], field_name: str) -> None:
    fields.add(field_name)


def _truncation_list(fields: set[str]) -> list[str]:
    return sorted(fields)


# ---------------------------------------------------------------------------
# Document containers and writer
# ---------------------------------------------------------------------------


@dataclass
class TypeStats:
    entity_type: str
    source_total: int = 0
    written: int = 0
    filtered: Counter = field(default_factory=Counter)
    truncated_docs: int = 0
    truncated_fields: Counter = field(default_factory=Counter)
    text: _TextStats = field(default_factory=_TextStats)
    missing_name_docs: int = 0
    multi_subject_docs: int = 0

    @property
    def filtered_total(self) -> int:
        return sum(self.filtered.values())

    def record_truncation(self, fields: list[str]) -> None:
        if not fields:
            return
        self.truncated_docs += 1
        for field_name in fields:
            self.truncated_fields[field_name] += 1


class _JsonlWriter:
    def __init__(self, final_path: Path) -> None:
        self.final_path = final_path
        self.temporary_path = final_path.with_name(final_path.name + ".tmp")
        self._handle = self.temporary_path.open("wb")
        self._hasher = hashlib.sha256()
        self.lines = 0
        self.bytes = 0

    def write(self, document: dict[str, Any]) -> None:
        data = (
            json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self._handle.write(data)
        self._hasher.update(data)
        self.lines += 1
        self.bytes += len(data)

    def finish(self) -> dict[str, Any]:
        self._handle.close()
        return {
            "path": str(self.final_path),
            "lines": self.lines,
            "bytes": self.bytes,
            "sha256": self._hasher.hexdigest(),
        }

    def commit(self) -> None:
        os.replace(self.temporary_path, self.final_path)

    def abort(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass
        if self.temporary_path.exists():
            self.temporary_path.unlink()


# ---------------------------------------------------------------------------
# Read-only DB helpers
# ---------------------------------------------------------------------------


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _count(connection: sqlite3.Connection, sql: str) -> int:
    return int(connection.execute(sql).fetchone()[0] or 0)


def _validate_schema(connection: sqlite3.Connection) -> None:
    try:
        row = connection.execute(
            "SELECT value FROM import_metadata WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(
            "数据库中缺少 import_metadata，请先运行 scripts/import_sqlite.py"
        ) from exc
    if row is None:
        raise RuntimeError("数据库缺少 schema_version，请先重新导入")
    try:
        version = int(json.loads(row[0]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("schema_version 不是有效整数") from exc
    if version != SCHEMA_VERSION:
        raise RuntimeError(
            f"数据库 schema_version={version}，文档构建器需要 {SCHEMA_VERSION}"
        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_subjects(
    connection: sqlite3.Connection,
    writer: _JsonlWriter,
    *,
    limit: int | None,
    seed: int,
) -> tuple[TypeStats, _Reservoir]:
    stats = TypeStats("subject")
    stats.source_total = _count(connection, "SELECT COUNT(*) FROM subject")
    sampler = _Reservoir(SAMPLE_SIZE_BY_TYPE["subject"], seed)

    for row in connection.execute(SUBJECT_SQL):
        if limit is not None and stats.written >= limit:
            break
        (
            entity_id,
            raw_name,
            raw_name_cn,
            infobox_raw,
            raw_summary,
            tags_json,
            type_code,
            platform_code,
            nsfw,
            score,
            rank,
            date,
        ) = row
        if not isinstance(raw_summary, str) or not raw_summary.strip():
            stats.filtered["empty_summary"] += 1
            continue
        name = _strip_text(raw_name)
        name_cn = _strip_text(raw_name_cn)
        if not name and not name_cn:
            stats.missing_name_docs += 1

        aliases, aliases_truncated, alias_count = _prepare_aliases(
            infobox_raw, exclude=(name, name_cn)
        )
        tags, tags_truncated = _tag_names(tags_json)
        summary, summary_truncated = _truncate_text(
            raw_summary, SUBJECT_SUMMARY_MAX_CHARS
        )

        truncated_fields: set[str] = set()
        if aliases_truncated:
            _mark_truncation(truncated_fields, "aliases")
        if tags_truncated:
            _mark_truncation(truncated_fields, "tags")
        if summary_truncated:
            _mark_truncation(truncated_fields, "summary")

        parts = []
        if name:
            parts.append(f"作品名称：{name}")
        if name_cn and name_cn != name:
            parts.append(f"中文名称：{name_cn}")
        if aliases:
            parts.append("别名：" + "、".join(aliases))
        if tags:
            parts.append("标签：" + "、".join(tags))
        if summary:
            parts.append("简介：" + summary)
        text, text_truncated = _compose_text(
            parts, TOTAL_TEXT_MAX_CHARS["subject"]
        )
        if text_truncated:
            _mark_truncation(truncated_fields, "text")

        document = {
            "doc_id": _doc_id("subject", entity_id),
            "entity_type": "subject",
            "entity_id": entity_id,
            "subject_id": entity_id,
            "source": MAIN_FILES["subject"],
            "text": text,
            "metadata": {
                "name": name,
                "name_cn": name_cn,
                "aliases": aliases,
                "alias_count": alias_count,
                "tags": tags,
                "type": type_code,
                "platform": platform_code,
                "nsfw": bool(nsfw) if nsfw is not None else None,
                "score": score,
                "rank": rank,
                "date": date,
                "truncated_fields": _truncation_list(truncated_fields),
                "doc_rules_version": RULES_VERSION,
            },
        }
        writer.write(document)
        stats.written += 1
        stats.text.add(len(text))
        stats.record_truncation(document["metadata"]["truncated_fields"])
        sampler.add(
            {
                "doc_id": document["doc_id"],
                "entity_type": "subject",
                "entity_id": entity_id,
                "subject_id": entity_id,
                "source": document["source"],
                "text": text,
            }
        )
    return stats, sampler


def _process_character_group(
    rows: list[tuple[Any, ...]],
    writer: _JsonlWriter,
    stats: TypeStats,
    sampler: _Reservoir,
) -> None:
    entity_id = rows[0][0]
    name = _strip_text(rows[0][1])
    infobox_raw = rows[0][2]
    raw_summary = rows[0][3]
    if not isinstance(raw_summary, str) or not raw_summary.strip():
        stats.filtered["empty_summary"] += 1
        return

    subjects: list[tuple[int, str]] = []
    seen_subjects: set[int] = set()
    for row in rows:
        subject_id = row[4]
        if not isinstance(subject_id, int) or subject_id in seen_subjects:
            continue
        subject_name = _strip_text(row[5]) or _strip_text(row[6])
        if not subject_name:
            subject_name = f"subject:{subject_id}"
        seen_subjects.add(subject_id)
        subjects.append((subject_id, subject_name))

    if not subjects:
        stats.filtered["no_subject"] += 1
        return

    summary, summary_truncated = _truncate_text(
        raw_summary, CHARACTER_SUMMARY_MAX_CHARS
    )
    aliases, aliases_truncated, alias_count = _prepare_aliases(
        infobox_raw, exclude=(name,)
    )
    truncated_fields: set[str] = set()
    if aliases_truncated:
        _mark_truncation(truncated_fields, "aliases")
    if summary_truncated:
        _mark_truncation(truncated_fields, "summary")
    if len(subjects) > MAX_SUBJECTS_IN_TEXT:
        _mark_truncation(truncated_fields, "subject_names")
    if len(subjects) > MAX_SUBJECTS_IN_METADATA:
        _mark_truncation(truncated_fields, "subject_ids")

    text_subjects = [subject_name for _, subject_name in subjects[:MAX_SUBJECTS_IN_TEXT]]
    parts = []
    if name:
        parts.append(f"角色名称：{name}")
    if aliases:
        parts.append("别名：" + "、".join(aliases))
    parts.append("所属作品：" + "、".join(text_subjects))
    if summary:
        parts.append("角色简介：" + summary)
    text, text_truncated = _compose_text(
        parts, TOTAL_TEXT_MAX_CHARS["character"]
    )
    if text_truncated:
        _mark_truncation(truncated_fields, "text")

    subject_ids = [subject_id for subject_id, _ in subjects]
    subject_names = [subject_name for _, subject_name in subjects]
    if not name:
        stats.missing_name_docs += 1
    if len(subjects) > 1:
        stats.multi_subject_docs += 1

    document = {
        "doc_id": _doc_id("character", entity_id),
        "entity_type": "character",
        "entity_id": entity_id,
        "subject_id": subject_ids[0],
        "source": MAIN_FILES["character"],
        "text": text,
        "metadata": {
            "name": name,
            "aliases": aliases,
            "alias_count": alias_count,
            "subject_id": subject_ids[0],
            "subject_ids": subject_ids[:MAX_SUBJECTS_IN_METADATA],
            "subject_names": subject_names[:MAX_SUBJECTS_IN_METADATA],
            "subject_count": len(subjects),
            "truncated_fields": _truncation_list(truncated_fields),
            "doc_rules_version": RULES_VERSION,
        },
    }
    writer.write(document)
    stats.written += 1
    stats.text.add(len(text))
    stats.record_truncation(document["metadata"]["truncated_fields"])
    sampler.add(
        {
            "doc_id": document["doc_id"],
            "entity_type": "character",
            "entity_id": entity_id,
            "subject_id": subject_ids[0],
            "source": document["source"],
            "text": text,
        }
    )


def _build_characters(
    connection: sqlite3.Connection,
    writer: _JsonlWriter,
    *,
    limit: int | None,
    seed: int,
) -> tuple[TypeStats, _Reservoir]:
    stats = TypeStats("character")
    stats.source_total = _count(connection, "SELECT COUNT(*) FROM character")
    sampler = _Reservoir(SAMPLE_SIZE_BY_TYPE["character"], seed)

    current_id: int | None = None
    group: list[tuple[Any, ...]] = []

    def flush() -> bool:
        nonlocal group
        if not group:
            return False
        _process_character_group(group, writer, stats, sampler)
        group = []
        return limit is not None and stats.written >= limit

    for row in connection.execute(CHARACTER_SQL):
        if current_id is None:
            current_id = row[0]
        elif row[0] != current_id:
            if flush():
                return stats, sampler
            current_id = row[0]
        group.append(row)
    flush()
    return stats, sampler


def _build_episodes(
    connection: sqlite3.Connection,
    writer: _JsonlWriter,
    *,
    limit: int | None,
    seed: int,
) -> tuple[TypeStats, _Reservoir]:
    stats = TypeStats("episode")
    stats.source_total = _count(connection, "SELECT COUNT(*) FROM episode")
    sampler = _Reservoir(SAMPLE_SIZE_BY_TYPE["episode"], seed)

    for row in connection.execute(EPISODE_SQL):
        if limit is not None and stats.written >= limit:
            break
        (
            entity_id,
            raw_name,
            raw_name_cn,
            raw_description,
            airdate,
            sort_value,
            subject_id,
            resolved_subject_id,
            raw_subject_name,
            raw_subject_name_cn,
        ) = row
        if not isinstance(raw_description, str) or not raw_description.strip():
            stats.filtered["empty_description"] += 1
            continue
        if not isinstance(resolved_subject_id, int):
            stats.filtered["orphan_subject"] += 1
            continue
        subject_id = resolved_subject_id
        category = classify_episode_description(raw_description)
        if category in EPISODE_INCLUDED_QUALITY:
            cleaned, removed_credit_lines = clean_episode_description(
                raw_description
            )
            if len(cleaned.strip()) < MIN_EPISODE_CLEANED_CHARS:
                stats.filtered["cleaned_too_short"] += 1
                continue
            description, description_truncated = _truncate_text(
                cleaned, EPISODE_DESCRIPTION_MAX_CHARS
            )
        else:
            if category == "credit_only":
                stats.filtered["credit_only"] += 1
            else:
                stats.filtered["too_short_or_unclear"] += 1
            continue

        name = _strip_text(raw_name)
        name_cn = _strip_text(raw_name_cn)
        subject_name = (
            _strip_text(raw_subject_name)
            or _strip_text(raw_subject_name_cn)
            or f"subject:{subject_id}"
        )
        truncated_fields: set[str] = set()
        if description_truncated:
            _mark_truncation(truncated_fields, "description")

        parts = [f"所属作品：{subject_name}"]
        if name:
            parts.append(f"章节名称：{name}")
        if name_cn and name_cn != name:
            parts.append(f"章节中文名：{name_cn}")
        if isinstance(airdate, str) and airdate.strip():
            parts.append(f"播出日期：{airdate.strip()}")
        parts.append("剧情：" + description)
        text, text_truncated = _compose_text(
            parts, TOTAL_TEXT_MAX_CHARS["episode"]
        )
        if text_truncated:
            _mark_truncation(truncated_fields, "text")
        if not name and not name_cn:
            stats.missing_name_docs += 1

        document = {
            "doc_id": _doc_id("episode", entity_id),
            "entity_type": "episode",
            "entity_id": entity_id,
            "subject_id": subject_id,
            "source": MAIN_FILES["episode"],
            "text": text,
            "metadata": {
                "name": name,
                "name_cn": name_cn,
                "subject_name": subject_name,
                "airdate": airdate,
                "sort": sort_value,
                "quality_category": category,
                "removed_credit_lines": removed_credit_lines,
                "truncated_fields": _truncation_list(truncated_fields),
                "doc_rules_version": RULES_VERSION,
            },
        }
        writer.write(document)
        stats.written += 1
        stats.text.add(len(text))
        stats.record_truncation(document["metadata"]["truncated_fields"])
        sampler.add(
            {
                "doc_id": document["doc_id"],
                "entity_type": "episode",
                "entity_id": entity_id,
                "subject_id": subject_id,
                "source": document["source"],
                "text": text,
            }
        )
    return stats, sampler


# ---------------------------------------------------------------------------
# Traceability and manifest
# ---------------------------------------------------------------------------


def _verify_samples(
    connection: sqlite3.Connection, samples: list[dict[str, Any]]
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    verified = 0
    for sample in samples:
        doc_id = sample["doc_id"]
        entity_type = sample["entity_type"]
        entity_id = sample["entity_id"]
        reason = None

        if entity_type == "subject":
            row = connection.execute(
                "SELECT id, name, name_cn FROM subject WHERE id = ?", (entity_id,)
            ).fetchone()
            if row is None:
                reason = "subject_missing"
            elif sample["subject_id"] != entity_id:
                reason = "subject_id_mismatch"
            else:
                text = sample["text"]
                if "简介：" not in text:
                    reason = "subject_template_mismatch"
                else:
                    names = [value for value in (row[1], row[2]) if value]
                    if names and not any(name in text for name in names):
                        reason = "subject_name_missing"
        elif entity_type == "character":
            row = connection.execute(
                "SELECT id, name FROM character WHERE id = ?", (entity_id,)
            ).fetchone()
            relation = connection.execute(
                "SELECT 1 FROM subject_characters WHERE character_id = ? LIMIT 1",
                (entity_id,),
            ).fetchone()
            subject = connection.execute(
                "SELECT 1 FROM subject WHERE id = ?", (sample["subject_id"],)
            ).fetchone()
            if row is None:
                reason = "character_missing"
            elif relation is None:
                reason = "character_relation_missing"
            elif subject is None:
                reason = "character_subject_missing"
            elif "角色简介：" not in sample["text"]:
                reason = "character_template_mismatch"
            elif row[1] and row[1] not in sample["text"]:
                reason = "character_name_missing"
        elif entity_type == "episode":
            row = connection.execute(
                """
                SELECT e.subject_id, e.name, e.name_cn,
                       s.name AS subject_name, s.name_cn AS subject_name_cn
                FROM episode e
                JOIN subject s ON s.id = e.subject_id
                WHERE e.id = ?
                """,
                (entity_id,),
            ).fetchone()
            if row is None:
                reason = "episode_missing"
            elif row[0] != sample["subject_id"]:
                reason = "episode_subject_mismatch"
            elif "剧情：" not in sample["text"]:
                reason = "episode_template_mismatch"
            else:
                subject_names = [value for value in (row[3], row[4]) if value]
                if subject_names and not any(name in sample["text"] for name in subject_names):
                    reason = "episode_subject_name_missing"
                elif row[1] and row[1] not in sample["text"]:
                    reason = "episode_name_missing"
        else:
            reason = "unknown_entity_type"

        if reason is None and not sample.get("text"):
            reason = "empty_text"
        if reason is None:
            verified += 1
        else:
            failures.append({"doc_id": doc_id, "reason": reason})

    return {
        "sample_size": len(samples),
        "verified": verified,
        "failures": failures,
    }


def _stats_dict(stats: TypeStats, *, partial: bool) -> dict[str, Any]:
    accounted = stats.written + stats.filtered_total
    return {
        "source_total": stats.source_total,
        "written": stats.written,
        "filtered": dict(sorted(stats.filtered.items())),
        "filtered_total": stats.filtered_total,
        "failed": 0,
        "accounted": accounted,
        "reconciled": None if partial else accounted == stats.source_total,
        "missing_name_docs": stats.missing_name_docs,
        "multi_subject_docs": stats.multi_subject_docs,
        "truncated_docs": stats.truncated_docs,
        "truncated_fields": dict(sorted(stats.truncated_fields.items())),
        "text_length": stats.text.summary(),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_build(
    *,
    db_path: Path = SQLITE_PATH,
    output_dir: Path = DOCUMENTS_DIR,
    limit: int | None = None,
    overwrite: bool = False,
    seed: int = 42,
    command: str = "python scripts/build_documents.py",
    verbose: bool = True,
) -> dict[str, Any]:
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    db_path = Path(db_path)
    output_dir = Path(output_dir)

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在：{db_path}")
    if db_path.is_dir():
        raise IsADirectoryError(f"数据库路径是目录：{db_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "subject": output_dir / "subjects.jsonl",
        "character": output_dir / "characters.jsonl",
        "episode": output_dir / "episodes.jsonl",
    }
    manifest_path = output_dir / "manifest.json"
    if not overwrite:
        existing = [
            path for path in list(final_paths.values()) + [manifest_path] if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "文档输出已存在："
                + ", ".join(str(path) for path in existing)
                + "。重建请显式传入 overwrite=True / --overwrite。"
            )

    for path in list(final_paths.values()) + [manifest_path]:
        stale = path.with_name(path.name + ".tmp")
        if stale.exists():
            stale.unlink()

    if verbose:
        print(f"数据库：{db_path}", file=sys.stderr, flush=True)
        print(
            f"模式：{'限量 ' + str(limit) + ' 篇/类' if limit is not None else '全量'}",
            file=sys.stderr,
            flush=True,
        )

    connection: sqlite3.Connection | None = None
    writers: list[_JsonlWriter] = []
    try:
        connection = _connect_readonly(db_path)
        _validate_schema(connection)

        writers = [_JsonlWriter(final_paths[key]) for key in ("subject", "character", "episode")]
        subject_writer, character_writer, episode_writer = writers

        if verbose:
            print("[文档] subject", file=sys.stderr, flush=True)
        subject_stats, subject_samples = _build_subjects(
            connection, subject_writer, limit=limit, seed=seed
        )
        if verbose:
            print(
                f"  subject   写入 {subject_stats.written:>9,}  "
                f"过滤 {subject_stats.filtered_total:>9,}",
                file=sys.stderr,
                flush=True,
            )

        if verbose:
            print("[文档] character", file=sys.stderr, flush=True)
        character_stats, character_samples = _build_characters(
            connection, character_writer, limit=limit, seed=seed + 1
        )
        if verbose:
            print(
                f"  character 写入 {character_stats.written:>9,}  "
                f"过滤 {character_stats.filtered_total:>9,}",
                file=sys.stderr,
                flush=True,
            )

        if verbose:
            print("[文档] episode", file=sys.stderr, flush=True)
        episode_stats, episode_samples = _build_episodes(
            connection, episode_writer, limit=limit, seed=seed + 2
        )
        if verbose:
            print(
                f"  episode   写入 {episode_stats.written:>9,}  "
                f"过滤 {episode_stats.filtered_total:>9,}",
                file=sys.stderr,
                flush=True,
            )

        all_stats = {
            "subject": subject_stats,
            "character": character_stats,
            "episode": episode_stats,
        }
        partial = limit is not None
        if not partial:
            for entity_type, stats in all_stats.items():
                if stats.written + stats.filtered_total != stats.source_total:
                    raise RuntimeError(
                        f"{entity_type} 文档数量对账失败："
                        f"{stats.written} + {stats.filtered_total} != {stats.source_total}"
                    )

        samples = (
            subject_samples.items + character_samples.items + episode_samples.items
        )
        traceability = _verify_samples(connection, samples)
        if traceability["failures"]:
            raise RuntimeError(
                "文档抽样溯源失败：" + json.dumps(traceability["failures"][:5], ensure_ascii=False)
            )

        file_info = {
            entity_type: writer.finish()
            for entity_type, writer in zip(("subject", "character", "episode"), writers)
        }
        for entity_type, info in file_info.items():
            if info["lines"] != all_stats[entity_type].written:
                raise RuntimeError(
                    f"{entity_type} 文件行数 {info['lines']} != 写入数 {all_stats[entity_type].written}"
                )

        for writer in writers:
            writer.commit()

        elapsed = time.perf_counter() - started
        manifest = {
            "generated_at": started_at.isoformat(timespec="seconds"),
            "command": command,
            "mode": "limit" if partial else "full",
            "limit": limit,
            "rules_version": RULES_VERSION,
            "schema_version": SCHEMA_VERSION,
            "db_path": str(db_path),
            "db_size_bytes": db_path.stat().st_size,
            "db_mtime_ns": db_path.stat().st_mtime_ns,
            "output_dir": str(output_dir),
            "elapsed_sec": round(elapsed, 3),
            "peak_rss_mb": peak_rss_mb(),
            "files": file_info,
            "counts": {
                key: _stats_dict(stats, partial=partial)
                for key, stats in all_stats.items()
            },
            "traceability": traceability,
            "sample_ids": {
                "subject": [item["doc_id"] for item in subject_samples.items],
                "character": [item["doc_id"] for item in character_samples.items],
                "episode": [item["doc_id"] for item in episode_samples.items],
            },
            "rules": {
                "subject_summary_max_chars": SUBJECT_SUMMARY_MAX_CHARS,
                "character_summary_max_chars": CHARACTER_SUMMARY_MAX_CHARS,
                "episode_description_max_chars": EPISODE_DESCRIPTION_MAX_CHARS,
                "total_text_max_chars": TOTAL_TEXT_MAX_CHARS,
                "max_aliases": MAX_ALIASES,
                "max_alias_chars": MAX_ALIAS_CHARS,
                "max_tags": MAX_TAGS,
                "max_tag_chars": MAX_TAG_CHARS,
                "max_subjects_in_text": MAX_SUBJECTS_IN_TEXT,
                "max_subjects_in_metadata": MAX_SUBJECTS_IN_METADATA,
                "episode_included_quality": sorted(EPISODE_INCLUDED_QUALITY),
                "min_episode_cleaned_chars": MIN_EPISODE_CLEANED_CHARS,
                "traceability_sample_by_type": SAMPLE_SIZE_BY_TYPE,
            },
            "notes": [
                "文档由只读 SQLite 数据库构建，原始实体 ID 与 source_file 保留。",
                "type/platform 等未解释码值只放 metadata，不写入检索文本。",
                "episode 仅纳入 narrative/mixed；mixed 文档会删除制作信息行。",
                "所有截断均记录在文档 metadata.truncated_fields，并在 manifest 汇总。",
                "doc_id 与输出顺序稳定，重复构建可得到相同 JSONL 内容。",
            ],
        }
        _write_manifest(manifest_path, manifest)

        if verbose:
            print(f"输出：{output_dir}", file=sys.stderr, flush=True)
            print(
                f"溯源抽样：{traceability['verified']}/{traceability['sample_size']}",
                file=sys.stderr,
                flush=True,
            )
            print(f"总耗时：{elapsed:.1f} 秒", file=sys.stderr, flush=True)

        return {
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "elapsed_sec": elapsed,
            "files": file_info,
            "counts": manifest["counts"],
            "traceability": traceability,
        }
    except Exception:
        for writer in writers:
            writer.abort()
        if connection is not None:
            connection.close()
        raise

    finally:
        if connection is not None:
            connection.close()
