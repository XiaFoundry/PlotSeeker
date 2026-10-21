"""Streaming JSONL reader shared by auditing and database import."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

MAX_ERROR_EXAMPLES = 20


@dataclass
class FileStats:
    name: str
    path: Path
    size_bytes: int = 0
    total_lines: int = 0
    valid_records: int = 0
    blank_lines: int = 0
    parse_errors: int = 0
    decode_errors: int = 0
    schema_errors: int = 0
    error_examples: list[dict[str, Any]] = field(default_factory=list)
    sha256: str = ""
    partial: bool = False
    elapsed_sec: float = 0.0

    @property
    def bad_lines(self) -> int:
        return self.parse_errors + self.decode_errors + self.schema_errors

    def add_error(self, lineno: int, kind: str, message: str, sample: str) -> None:
        if len(self.error_examples) >= MAX_ERROR_EXAMPLES:
            return
        self.error_examples.append(
            {
                "line": lineno,
                "kind": kind,
                "error": message[:300],
                "sample": sample[:300].replace("\n", "\\n"),
            }
        )


def iter_jsonl(
    path: Path,
    stats: FileStats,
    limit: int | None = None,
    hash_file: bool = True,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, record)`` while updating ``stats`` in place."""

    stats.path = path
    stats.name = path.name
    stats.size_bytes = path.stat().st_size
    stats.partial = limit is not None
    hasher = hashlib.sha256() if (hash_file and limit is None) else None

    with path.open("rb") as handle:
        for lineno, raw in enumerate(handle, 1):
            if limit is not None and stats.total_lines >= limit:
                break
            stats.total_lines += 1
            if hasher is not None:
                hasher.update(raw)

            stripped = raw.strip()
            if not stripped:
                stats.blank_lines += 1
                continue

            try:
                text = stripped.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                stats.decode_errors += 1
                stats.add_error(lineno, "decode", str(exc), stripped[:200].decode("utf-8", "replace"))
                continue

            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                stats.parse_errors += 1
                stats.add_error(lineno, "json", str(exc), text)
                continue

            if not isinstance(record, dict):
                stats.schema_errors += 1
                stats.add_error(lineno, "schema", f"expected object, got {type(record).__name__}", text)
                continue

            stats.valid_records += 1
            yield lineno, record

    if hasher is not None:
        stats.sha256 = hasher.hexdigest()
