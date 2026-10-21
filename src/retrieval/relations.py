"""Map character/episode hits to subject candidates through P2 relations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from config import SQLITE_PATH
from retrieval.models import DocumentHit

MAX_SECONDARY_SUBJECTS = 4
SECONDARY_RELATION_WEIGHT = 0.35


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


class SubjectResolver:
    def __init__(self, db_path: Path = SQLITE_PATH) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.is_file():
            raise FileNotFoundError(f"SQLite 数据库不存在：{self.db_path}")
        self._connection = sqlite3.connect(
            f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False
        )
        self._cache: dict[int, dict[str, Any] | None] = {}

    def subject(self, subject_id: int) -> dict[str, Any] | None:
        if subject_id in self._cache:
            return self._cache[subject_id]
        row = self._connection.execute(
            "SELECT id, name, name_cn, type, platform, date, score, nsfw "
            "FROM subject WHERE id=?",
            (int(subject_id),),
        ).fetchone()
        result = None
        if row is not None:
            result = {
                "subject_id": int(row[0]),
                "name": row[1] or "",
                "name_cn": row[2] or "",
                "type": row[3],
                "platform": row[4],
                "date": row[5],
                "score": row[6],
                "nsfw": bool(row[7]),
            }
        self._cache[subject_id] = result
        return result

    def resolve_hit(self, hit: DocumentHit) -> list[tuple[int, str, float]]:
        if hit.entity_type == "subject":
            if hit.subject_id > 0:
                return [(hit.subject_id, f"subject:{hit.subject_id}", 1.0)]
            if hit.entity_id > 0:
                return [(hit.entity_id, f"subject:{hit.entity_id}", 1.0)]
            return []
        primary = int(hit.subject_id)
        if primary <= 0:
            return []
        resolved: list[tuple[int, str, float]] = [
            (primary, f"{hit.entity_type}:{hit.entity_id}->subject:{primary}", 1.0)
        ]
        if hit.entity_type != "character":
            return resolved
        seen = {primary}
        secondary = 0
        for value in _as_list(hit.metadata.get("subject_ids")):
            try:
                subject_id = int(value)
            except (TypeError, ValueError):
                continue
            if subject_id <= 0 or subject_id in seen:
                continue
            seen.add(subject_id)
            resolved.append(
                (
                    subject_id,
                    f"character:{hit.entity_id}->subject:{subject_id}",
                    SECONDARY_RELATION_WEIGHT,
                )
            )
            secondary += 1
            if secondary >= MAX_SECONDARY_SUBJECTS:
                break
        return resolved

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SubjectResolver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
