"""Import the Bangumi Archive JSONL files into a structured SQLite database.

Design notes
------------
* The import streams every JSONL file and batches inserts inside one
  transaction per table. It never loads a whole file into memory.
* The database is always built into a temporary file and atomically replaced
  only after a successful import, so an existing database is never left in a
  half-written state. Replacing an existing database requires ``overwrite=True``.
* Hard foreign keys are not declared because the archive contains measurable
  dangling references (deleted subjects, missing people, etc.). Equivalent
  integrity checks are run after import and recorded in the import report.
  ``PRAGMA foreign_keys`` is still enabled and verified, so future schema
  additions can rely on it.
* In ``--limit`` mode, relation rows are imported only when both endpoints were
  imported from the same limited main-entity scan. Filtered rows are reported
  separately and are not counted as orphans.
* ``person_relations`` uses ``person_type`` values ``crt`` / ``prsn`` and the
  person/character ID spaces overlap heavily, so it is checked by namespace
  membership rather than a single foreign-key target.
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config import (
    IMPORT_REPORT_PATH,
    MAIN_ENTITY_FILES,
    RAW_DIR,
    RELATION_FILES,
    SQLITE_PATH,
    STORAGE_DIR,
)
from ingestion.jsonl import FileStats, iter_jsonl
from ingestion.resources import peak_rss_mb

SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZE = 5000
TABLE_ORDER = [
    "subject",
    "character",
    "person",
    "episode",
    "subject_characters",
    "subject_persons",
    "subject_relations",
    "person_characters",
    "person_relations",
]

SCHEMA_SQL = """
CREATE TABLE subject (
    id INTEGER PRIMARY KEY,
    type INTEGER,
    name TEXT,
    name_cn TEXT,
    infobox_raw TEXT,
    platform INTEGER,
    summary TEXT,
    nsfw INTEGER,
    tags_json TEXT,
    meta_tags_json TEXT,
    score REAL,
    score_details_json TEXT,
    rank INTEGER,
    date TEXT,
    favorite_json TEXT,
    series INTEGER,
    source_file TEXT NOT NULL
);

CREATE TABLE character (
    id INTEGER PRIMARY KEY,
    role INTEGER,
    name TEXT,
    infobox_raw TEXT,
    summary TEXT,
    comments INTEGER,
    collects INTEGER,
    source_file TEXT NOT NULL
);

CREATE TABLE person (
    id INTEGER PRIMARY KEY,
    name TEXT,
    type INTEGER,
    career_json TEXT,
    infobox_raw TEXT,
    summary TEXT,
    comments INTEGER,
    collects INTEGER,
    source_file TEXT NOT NULL
);

CREATE TABLE episode (
    id INTEGER PRIMARY KEY,
    name TEXT,
    name_cn TEXT,
    description TEXT,
    airdate TEXT,
    disc INTEGER,
    duration TEXT,
    subject_id INTEGER,
    sort REAL,
    type INTEGER,
    source_file TEXT NOT NULL
);

CREATE TABLE subject_characters (
    subject_id INTEGER NOT NULL,
    character_id INTEGER NOT NULL,
    type INTEGER,
    ord INTEGER,
    PRIMARY KEY (subject_id, character_id)
) WITHOUT ROWID;

CREATE TABLE subject_persons (
    subject_id INTEGER NOT NULL,
    person_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    appear_eps TEXT,
    PRIMARY KEY (subject_id, person_id, position)
) WITHOUT ROWID;

CREATE TABLE subject_relations (
    subject_id INTEGER NOT NULL,
    related_subject_id INTEGER NOT NULL,
    relation_type INTEGER NOT NULL,
    ord INTEGER,
    PRIMARY KEY (subject_id, related_subject_id, relation_type)
) WITHOUT ROWID;

CREATE TABLE person_characters (
    person_id INTEGER NOT NULL,
    subject_id INTEGER NOT NULL,
    character_id INTEGER NOT NULL,
    type INTEGER,
    summary TEXT,
    PRIMARY KEY (person_id, subject_id, character_id)
) WITHOUT ROWID;

CREATE TABLE person_relations (
    person_type TEXT NOT NULL,
    person_id INTEGER NOT NULL,
    related_person_id INTEGER NOT NULL,
    relation_type INTEGER NOT NULL,
    spoiler INTEGER,
    ended INTEGER,
    PRIMARY KEY (person_type, person_id, related_person_id, relation_type)
) WITHOUT ROWID;

CREATE TABLE import_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

INDEX_SQL = [
    "CREATE INDEX idx_subject_name ON subject(name)",
    "CREATE INDEX idx_subject_name_cn ON subject(name_cn)",
    "CREATE INDEX idx_subject_type ON subject(type)",
    "CREATE INDEX idx_subject_platform ON subject(platform)",
    "CREATE INDEX idx_episode_subject_id ON episode(subject_id)",
    "CREATE INDEX idx_episode_name ON episode(name)",
    "CREATE INDEX idx_episode_name_cn ON episode(name_cn)",
    "CREATE INDEX idx_character_name ON character(name)",
    "CREATE INDEX idx_person_name ON person(name)",
    "CREATE INDEX idx_subject_characters_character ON subject_characters(character_id)",
    "CREATE INDEX idx_subject_persons_person ON subject_persons(person_id)",
    "CREATE INDEX idx_subject_relations_related ON subject_relations(related_subject_id)",
    "CREATE INDEX idx_person_characters_character ON person_characters(character_id)",
    "CREATE INDEX idx_person_characters_subject ON person_characters(subject_id)",
    "CREATE INDEX idx_person_relations_related ON person_relations(related_person_id)",
]


@dataclass
class FileImportResult:
    file: str
    table: str
    valid_records: int = 0
    inserted: int = 0
    invalid_rows: int = 0
    filtered_by_limit: int = 0
    ignored: int = 0
    sha256: str = ""
    partial: bool = False
    bad_lines: int = 0
    elapsed_sec: float = 0.0
    error_examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accounted(self) -> int:
        return self.invalid_rows + self.filtered_by_limit + self.inserted + self.ignored

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "table": self.table,
            "valid_records": self.valid_records,
            "inserted": self.inserted,
            "invalid_rows": self.invalid_rows,
            "filtered_by_limit": self.filtered_by_limit,
            "ignored": self.ignored,
            "accounted": self.accounted,
            "reconciled": self.accounted == self.valid_records,
            "sha256": self.sha256,
            "partial": self.partial,
            "bad_lines": self.bad_lines,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "error_examples": self.error_examples,
        }


# ---------------------------------------------------------------------------
# Value conversion helpers
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _int(value: Any) -> int | None:
    return value if _is_int(value) else None


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _bool_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if _is_int(value):
        return int(bool(value))
    return None


def _json_text(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _subject_row(record: dict[str, Any], source_file: str) -> tuple[Any, ...]:
    return (
        record.get("id"),
        _int(record.get("type")),
        _text(record.get("name")),
        _text(record.get("name_cn")),
        _text(record.get("infobox")),
        _int(record.get("platform")),
        _text(record.get("summary")),
        _bool_int(record.get("nsfw")),
        _json_text(record.get("tags")),
        _json_text(record.get("meta_tags")),
        _float(record.get("score")),
        _json_text(record.get("score_details")),
        _int(record.get("rank")),
        _text(record.get("date")),
        _json_text(record.get("favorite")),
        _bool_int(record.get("series")),
        source_file,
    )


def _character_row(record: dict[str, Any], source_file: str) -> tuple[Any, ...]:
    return (
        record.get("id"),
        _int(record.get("role")),
        _text(record.get("name")),
        _text(record.get("infobox")),
        _text(record.get("summary")),
        _int(record.get("comments")),
        _int(record.get("collects")),
        source_file,
    )


def _person_row(record: dict[str, Any], source_file: str) -> tuple[Any, ...]:
    return (
        record.get("id"),
        _text(record.get("name")),
        _int(record.get("type")),
        _json_text(record.get("career")),
        _text(record.get("infobox")),
        _text(record.get("summary")),
        _int(record.get("comments")),
        _int(record.get("collects")),
        source_file,
    )


def _episode_row(record: dict[str, Any], source_file: str) -> tuple[Any, ...]:
    return (
        record.get("id"),
        _text(record.get("name")),
        _text(record.get("name_cn")),
        _text(record.get("description")),
        _text(record.get("airdate")),
        _int(record.get("disc")),
        _text(record.get("duration")),
        _int(record.get("subject_id")),
        _float(record.get("sort")),
        _int(record.get("type")),
        source_file,
    )


MAIN_IMPORT_SPECS: dict[str, dict[str, Any]] = {
    "subject": {
        "file": MAIN_ENTITY_FILES["subject"],
        "table": "subject",
        "columns": (
            "id",
            "type",
            "name",
            "name_cn",
            "infobox_raw",
            "platform",
            "summary",
            "nsfw",
            "tags_json",
            "meta_tags_json",
            "score",
            "score_details_json",
            "rank",
            "date",
            "favorite_json",
            "series",
            "source_file",
        ),
        "row_builder": _subject_row,
    },
    "character": {
        "file": MAIN_ENTITY_FILES["character"],
        "table": "character",
        "columns": (
            "id",
            "role",
            "name",
            "infobox_raw",
            "summary",
            "comments",
            "collects",
            "source_file",
        ),
        "row_builder": _character_row,
    },
    "person": {
        "file": MAIN_ENTITY_FILES["person"],
        "table": "person",
        "columns": (
            "id",
            "name",
            "type",
            "career_json",
            "infobox_raw",
            "summary",
            "comments",
            "collects",
            "source_file",
        ),
        "row_builder": _person_row,
    },
    "episode": {
        "file": MAIN_ENTITY_FILES["episode"],
        "table": "episode",
        "columns": (
            "id",
            "name",
            "name_cn",
            "description",
            "airdate",
            "disc",
            "duration",
            "subject_id",
            "sort",
            "type",
            "source_file",
        ),
        "row_builder": _episode_row,
    },
}


def _subject_characters_row(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("subject_id"),
        record.get("character_id"),
        _int(record.get("type")),
        _int(record.get("order")),
    )


def _subject_persons_row(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("subject_id"),
        record.get("person_id"),
        record.get("position"),
        _text(record.get("appear_eps")),
    )


def _subject_relations_row(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("subject_id"),
        record.get("related_subject_id"),
        record.get("relation_type"),
        _int(record.get("order")),
    )


def _person_characters_row(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("person_id"),
        record.get("subject_id"),
        record.get("character_id"),
        _int(record.get("type")),
        _text(record.get("summary")),
    )


def _person_relations_row(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("person_type"),
        record.get("person_id"),
        record.get("related_person_id"),
        record.get("relation_type"),
        _bool_int(record.get("spoiler")),
        _bool_int(record.get("ended")),
    )


RELATION_IMPORT_SPECS: dict[str, dict[str, Any]] = {
    "subject-characters": {
        "file": RELATION_FILES["subject-characters"],
        "table": "subject_characters",
        "columns": ("subject_id", "character_id", "type", "ord"),
        "key_fields": ("subject_id", "character_id"),
        "row_builder": _subject_characters_row,
    },
    "subject-persons": {
        "file": RELATION_FILES["subject-persons"],
        "table": "subject_persons",
        "columns": ("subject_id", "person_id", "position", "appear_eps"),
        "key_fields": ("subject_id", "person_id", "position"),
        "row_builder": _subject_persons_row,
    },
    "subject-relations": {
        "file": RELATION_FILES["subject-relations"],
        "table": "subject_relations",
        "columns": ("subject_id", "related_subject_id", "relation_type", "ord"),
        "key_fields": ("subject_id", "related_subject_id", "relation_type"),
        "row_builder": _subject_relations_row,
    },
    "person-characters": {
        "file": RELATION_FILES["person-characters"],
        "table": "person_characters",
        "columns": ("person_id", "subject_id", "character_id", "type", "summary"),
        "key_fields": ("person_id", "subject_id", "character_id"),
        "row_builder": _person_characters_row,
    },
    "person-relations": {
        "file": RELATION_FILES["person-relations"],
        "table": "person_relations",
        "columns": (
            "person_type",
            "person_id",
            "related_person_id",
            "relation_type",
            "spoiler",
            "ended",
        ),
        "key_fields": ("person_id", "related_person_id", "relation_type"),
        "row_builder": _person_relations_row,
    },
}

INSERT_SQL = {
    spec["table"]: (
        f"INSERT OR IGNORE INTO {spec['table']} "
        f"({', '.join(spec['columns'])}) "
        f"VALUES ({', '.join('?' for _ in spec['columns'])})"
    )
    for spec in list(MAIN_IMPORT_SPECS.values()) + list(RELATION_IMPORT_SPECS.values())
}


# ---------------------------------------------------------------------------
# Import steps
# ---------------------------------------------------------------------------


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -262144")
    conn.execute("PRAGMA locking_mode = EXCLUSIVE")
    conn.execute("PRAGMA foreign_keys = ON")
    if int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
        raise RuntimeError("SQLite foreign_keys pragma could not be enabled")


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def _import_main(
    conn: sqlite3.Connection,
    entity: str,
    spec: dict[str, Any],
    raw_dir: Path,
    *,
    limit: int | None,
    batch_size: int,
    id_sets: dict[str, set[int]] | None,
    verbose: bool,
) -> FileImportResult:
    stats = FileStats(name=spec["file"], path=raw_dir / spec["file"])
    sql = INSERT_SQL[spec["table"]]
    row_builder: Callable[[dict[str, Any], str], tuple[Any, ...]] = spec["row_builder"]
    result = FileImportResult(file=spec["file"], table=spec["table"])
    target_ids = id_sets.get(entity) if id_sets is not None else None
    batch: list[tuple[Any, ...]] = []
    before_changes = conn.total_changes
    started = time.perf_counter()

    for _, record in iter_jsonl(raw_dir / spec["file"], stats, limit=limit):
        record_id = record.get("id")
        if not _is_int(record_id):
            result.invalid_rows += 1
            continue
        if entity == "episode" and id_sets is not None:
            subject_id = record.get("subject_id")
            if _is_int(subject_id) and subject_id not in id_sets["subject"]:
                result.filtered_by_limit += 1
                continue
        if target_ids is not None:
            target_ids.add(record_id)
        batch.append(row_builder(record, spec["file"]))
        if len(batch) >= batch_size:
            conn.executemany(sql, batch)
            batch.clear()

    if batch:
        conn.executemany(sql, batch)
    conn.commit()

    result.valid_records = stats.valid_records
    result.inserted = conn.total_changes - before_changes
    result.ignored = (
        result.valid_records
        - result.invalid_rows
        - result.filtered_by_limit
        - result.inserted
    )
    result.sha256 = stats.sha256
    result.partial = stats.partial
    result.bad_lines = stats.bad_lines
    result.error_examples = stats.error_examples
    result.elapsed_sec = time.perf_counter() - started

    if verbose:
        print(
            f"  {entity:<10} {result.valid_records:>9,} 行  "
            f"写入 {result.inserted:>9,} 行  "
            f"限量过滤 {result.filtered_by_limit:>6,}  {result.elapsed_sec:>6.1f}s",
            file=sys.stderr,
            flush=True,
        )
    return result


def _valid_relation_row(spec: dict[str, Any], record: dict[str, Any]) -> bool:
    for field_name in spec["key_fields"]:
        if not _is_int(record.get(field_name)):
            return False
    if spec["table"] == "person_relations":
        person_type = record.get("person_type")
        if not isinstance(person_type, str) or not person_type.strip():
            return False
    return True


def _relation_eligible(
    spec: dict[str, Any],
    record: dict[str, Any],
    id_sets: dict[str, set[int]],
) -> bool:
    subjects = id_sets["subject"]
    characters = id_sets["character"]
    persons = id_sets["person"]
    table = spec["table"]

    if table == "subject_characters":
        return record["subject_id"] in subjects and record["character_id"] in characters
    if table == "subject_persons":
        return record["subject_id"] in subjects and record["person_id"] in persons
    if table == "subject_relations":
        return record["subject_id"] in subjects and record["related_subject_id"] in subjects
    if table == "person_characters":
        return (
            record["person_id"] in persons
            and record["subject_id"] in subjects
            and record["character_id"] in characters
        )
    if table == "person_relations":
        left = record["person_id"]
        right = record["related_person_id"]
        person_type = record.get("person_type")
        if person_type == "prsn":
            return left in persons and right in persons
        if person_type == "crt":
            return left in characters and right in characters
        return (left in persons and right in persons) or (left in characters and right in characters)
    return False


def _import_relation(
    conn: sqlite3.Connection,
    name: str,
    spec: dict[str, Any],
    raw_dir: Path,
    *,
    limit: int | None,
    batch_size: int,
    id_sets: dict[str, set[int]] | None,
    verbose: bool,
) -> FileImportResult:
    stats = FileStats(name=spec["file"], path=raw_dir / spec["file"])
    sql = INSERT_SQL[spec["table"]]
    row_builder: Callable[[dict[str, Any]], tuple[Any, ...]] = spec["row_builder"]
    result = FileImportResult(file=spec["file"], table=spec["table"])
    batch: list[tuple[Any, ...]] = []
    before_changes = conn.total_changes
    started = time.perf_counter()

    for _, record in iter_jsonl(raw_dir / spec["file"], stats, limit=limit):
        if not _valid_relation_row(spec, record):
            result.invalid_rows += 1
            continue
        if id_sets is not None and not _relation_eligible(spec, record, id_sets):
            result.filtered_by_limit += 1
            continue
        batch.append(row_builder(record))
        if len(batch) >= batch_size:
            conn.executemany(sql, batch)
            batch.clear()

    if batch:
        conn.executemany(sql, batch)
    conn.commit()

    result.valid_records = stats.valid_records
    result.inserted = conn.total_changes - before_changes
    result.ignored = (
        result.valid_records
        - result.invalid_rows
        - result.filtered_by_limit
        - result.inserted
    )
    result.sha256 = stats.sha256
    result.partial = stats.partial
    result.bad_lines = stats.bad_lines
    result.error_examples = stats.error_examples
    result.elapsed_sec = time.perf_counter() - started

    if verbose:
        print(
            f"  {name:<19} {result.valid_records:>9,} 行  "
            f"写入 {result.inserted:>9,} 行  "
            f"限量过滤 {result.filtered_by_limit:>6,}  {result.elapsed_sec:>6.1f}s",
            file=sys.stderr,
            flush=True,
        )
    return result


# ---------------------------------------------------------------------------
# Integrity checks and report
# ---------------------------------------------------------------------------


def _scalar(conn: sqlite3.Connection, sql: str) -> int:
    value = conn.execute(sql).fetchone()[0]
    return int(value or 0)


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: _scalar(conn, f"SELECT COUNT(*) FROM {table}")
        for table in TABLE_ORDER
    }


def _primary_key_duplicates(conn: sqlite3.Connection) -> dict[str, int]:
    keys = {
        "subject": "id",
        "character": "id",
        "person": "id",
        "episode": "id",
        "subject_characters": "subject_id, character_id",
        "subject_persons": "subject_id, person_id, position",
        "subject_relations": "subject_id, related_subject_id, relation_type",
        "person_characters": "person_id, subject_id, character_id",
        "person_relations": "person_type, person_id, related_person_id, relation_type",
    }
    result = {}
    for table, columns in keys.items():
        result[table] = _scalar(
            conn,
            f"SELECT COUNT(*) FROM (SELECT {columns} FROM {table} "
            f"GROUP BY {columns} HAVING COUNT(*) > 1)",
        )
    return result


def _orphans(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "episode.subject_id": _scalar(
            conn,
            "SELECT COUNT(*) FROM episode e LEFT JOIN subject s ON s.id = e.subject_id "
            "WHERE e.subject_id IS NOT NULL AND s.id IS NULL",
        ),
        "episode.subject_id_null": _scalar(
            conn, "SELECT COUNT(*) FROM episode WHERE subject_id IS NULL"
        ),
        "subject_characters.subject_id": _scalar(
            conn,
            "SELECT COUNT(*) FROM subject_characters sc "
            "LEFT JOIN subject s ON s.id = sc.subject_id WHERE s.id IS NULL",
        ),
        "subject_characters.character_id": _scalar(
            conn,
            "SELECT COUNT(*) FROM subject_characters sc "
            "LEFT JOIN character c ON c.id = sc.character_id WHERE c.id IS NULL",
        ),
        "subject_persons.subject_id": _scalar(
            conn,
            "SELECT COUNT(*) FROM subject_persons sp "
            "LEFT JOIN subject s ON s.id = sp.subject_id WHERE s.id IS NULL",
        ),
        "subject_persons.person_id": _scalar(
            conn,
            "SELECT COUNT(*) FROM subject_persons sp "
            "LEFT JOIN person p ON p.id = sp.person_id WHERE p.id IS NULL",
        ),
        "subject_relations.subject_id": _scalar(
            conn,
            "SELECT COUNT(*) FROM subject_relations sr "
            "LEFT JOIN subject s ON s.id = sr.subject_id WHERE s.id IS NULL",
        ),
        "subject_relations.related_subject_id": _scalar(
            conn,
            "SELECT COUNT(*) FROM subject_relations sr "
            "LEFT JOIN subject s ON s.id = sr.related_subject_id WHERE s.id IS NULL",
        ),
        "subject_relations.self_loop": _scalar(
            conn, "SELECT COUNT(*) FROM subject_relations WHERE subject_id = related_subject_id"
        ),
        "person_characters.person_id": _scalar(
            conn,
            "SELECT COUNT(*) FROM person_characters pc "
            "LEFT JOIN person p ON p.id = pc.person_id WHERE p.id IS NULL",
        ),
        "person_characters.subject_id": _scalar(
            conn,
            "SELECT COUNT(*) FROM person_characters pc "
            "LEFT JOIN subject s ON s.id = pc.subject_id WHERE s.id IS NULL",
        ),
        "person_characters.character_id": _scalar(
            conn,
            "SELECT COUNT(*) FROM person_characters pc "
            "LEFT JOIN character c ON c.id = pc.character_id WHERE c.id IS NULL",
        ),
        "person_relations.person_id_unresolved": _scalar(
            conn,
            "SELECT COUNT(*) FROM person_relations pr "
            "LEFT JOIN person p ON p.id = pr.person_id "
            "LEFT JOIN character c ON c.id = pr.person_id "
            "WHERE p.id IS NULL AND c.id IS NULL",
        ),
        "person_relations.related_person_id_unresolved": _scalar(
            conn,
            "SELECT COUNT(*) FROM person_relations pr "
            "LEFT JOIN person p ON p.id = pr.related_person_id "
            "LEFT JOIN character c ON c.id = pr.related_person_id "
            "WHERE p.id IS NULL AND c.id IS NULL",
        ),
        "person_relations.unknown_person_type": _scalar(
            conn,
            "SELECT COUNT(*) FROM person_relations WHERE person_type NOT IN ('crt', 'prsn')",
        ),
    }


def _person_relation_membership(
    conn: sqlite3.Connection, field_name: str
) -> dict[str, dict[str, int]]:
    if field_name not in ("person_id", "related_person_id"):
        raise ValueError(f"unexpected field: {field_name}")
    rows = conn.execute(
        f"""
        SELECT pr.person_type,
               SUM(CASE WHEN p.id IS NOT NULL AND c.id IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN p.id IS NULL AND c.id IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN p.id IS NOT NULL AND c.id IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN p.id IS NULL AND c.id IS NULL THEN 1 ELSE 0 END),
               COUNT(*)
        FROM person_relations pr
        LEFT JOIN person p ON p.id = pr.{field_name}
        LEFT JOIN character c ON c.id = pr.{field_name}
        GROUP BY pr.person_type
        ORDER BY pr.person_type
        """
    ).fetchall()
    result = {}
    for person_type, person_only, character_only, both, neither, total in rows:
        result[person_type] = {
            "person_only": int(person_only or 0),
            "character_only": int(character_only or 0),
            "both": int(both or 0),
            "neither": int(neither or 0),
            "total": int(total or 0),
        }
    return result


def _integrity(conn: sqlite3.Connection) -> dict[str, Any]:
    integrity_rows = conn.execute("PRAGMA integrity_check").fetchall()
    integrity_text = "; ".join(str(row[0]) for row in integrity_rows[:5])
    foreign_key_rows = conn.execute("PRAGMA foreign_key_check").fetchall()
    return {
        "pragma_foreign_keys": int(conn.execute("PRAGMA foreign_keys").fetchone()[0]),
        "pragma_integrity_check": integrity_text,
        "foreign_key_violations": len(foreign_key_rows),
        "foreign_key_note": (
            "原始归档存在悬空引用，因此未声明硬外键；"
            "改用导入后完整性查询，PRAGMA foreign_keys 仍已启用。"
        ),
        "primary_key_duplicates": _primary_key_duplicates(conn),
        "orphans": _orphans(conn),
        "person_relation_membership": {
            "person_id": _person_relation_membership(conn, "person_id"),
            "related_person_id": _person_relation_membership(conn, "related_person_id"),
        },
    }


def _query(conn: sqlite3.Connection, sql: str, limit: int = 3) -> list[dict[str, Any]]:
    cursor = conn.execute(sql)
    columns = [description[0] for description in cursor.description or []]
    return [dict(zip(columns, row)) for row in cursor.fetchmany(limit)]


def _samples(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    return {
        "character_to_subject": _query(
            conn,
            """
            SELECT sc.character_id, c.name AS character_name,
                   sc.subject_id, s.name AS subject_name
            FROM subject_characters sc
            JOIN character c ON c.id = sc.character_id
            JOIN subject s ON s.id = sc.subject_id
            ORDER BY sc.subject_id
            LIMIT 3
            """,
        ),
        "episode_to_subject": _query(
            conn,
            """
            SELECT e.id AS episode_id, e.name AS episode_name,
                   e.subject_id, s.name AS subject_name
            FROM episode e
            JOIN subject s ON s.id = e.subject_id
            WHERE e.subject_id IS NOT NULL
            ORDER BY e.id
            LIMIT 3
            """,
        ),
        "subject_relations": _query(
            conn,
            """
            SELECT sr.subject_id, s1.name AS subject_name,
                   sr.relation_type, sr.related_subject_id, s2.name AS related_name
            FROM subject_relations sr
            JOIN subject s1 ON s1.id = sr.subject_id
            JOIN subject s2 ON s2.id = sr.related_subject_id
            ORDER BY sr.subject_id
            LIMIT 3
            """,
        ),
        "person_character": _query(
            conn,
            """
            SELECT pc.person_id, p.name AS person_name,
                   pc.character_id, c.name AS character_name,
                   pc.subject_id, s.name AS subject_name
            FROM person_characters pc
            JOIN person p ON p.id = pc.person_id
            JOIN character c ON c.id = pc.character_id
            JOIN subject s ON s.id = pc.subject_id
            ORDER BY pc.person_id
            LIMIT 3
            """,
        ),
    }


def _write_metadata(
    conn: sqlite3.Connection,
    *,
    raw_dir: Path,
    mode: str,
    limit: int | None,
    report_path: Path,
    files: dict[str, FileImportResult],
    table_counts: dict[str, int],
    integrity: dict[str, Any],
    elapsed_sec: float,
) -> None:
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "limit": limit,
        "raw_dir": str(raw_dir),
        "import_report_path": str(report_path),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "sqlite_version": sqlite3.sqlite_version,
        "elapsed_sec": round(elapsed_sec, 3),
        "table_counts": table_counts,
        "integrity_check": integrity["pragma_integrity_check"],
        "foreign_key_violations": integrity["foreign_key_violations"],
        "source_files": {
            result.file: {
                "sha256": result.sha256,
                "partial": result.partial,
                "valid_records": result.valid_records,
                "inserted": result.inserted,
                "invalid_rows": result.invalid_rows,
                "filtered_by_limit": result.filtered_by_limit,
                "ignored": result.ignored,
            }
            for result in files.values()
        },
    }
    conn.executemany(
        "INSERT OR REPLACE INTO import_metadata(key, value) VALUES (?, ?)",
        [(key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()],
    )
    conn.commit()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_import(
    *,
    raw_dir: Path = RAW_DIR,
    db_path: Path = SQLITE_PATH,
    report_path: Path | None = None,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
    command: str = "python scripts/import_sqlite.py",
    verbose: bool = True,
) -> dict[str, Any]:
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    raw_dir = Path(raw_dir)
    db_path = Path(db_path)
    if report_path is None:
        report_path = (
            STORAGE_DIR / "import-report-smoke.json"
            if limit is not None
            else IMPORT_REPORT_PATH
        )
    report_path = Path(report_path)

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    missing_files = [
        filename
        for filename in list(MAIN_ENTITY_FILES.values()) + list(RELATION_FILES.values())
        if not (raw_dir / filename).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "缺少原始数据文件：" + ", ".join(str(raw_dir / name) for name in missing_files)
        )

    if db_path.exists() and db_path.is_dir():
        raise IsADirectoryError(f"数据库路径是目录：{db_path}")
    if db_path.exists() and not overwrite:
        raise FileExistsError(
            f"数据库已存在：{db_path}。重建请显式传入 overwrite=True / --overwrite。"
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_db = db_path.with_name(db_path.name + ".tmp")
    if temporary_db.exists():
        temporary_db.unlink()

    if verbose:
        print(f"原始数据：{raw_dir}", file=sys.stderr, flush=True)
        print(f"目标数据库：{db_path}", file=sys.stderr, flush=True)
        print(f"模式：{'限量 ' + str(limit) if limit is not None else '全量'}", file=sys.stderr, flush=True)

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(temporary_db)
        _configure_connection(conn)
        _create_schema(conn)

        id_sets: dict[str, set[int]] | None = (
            {"subject": set(), "character": set(), "person": set()}
            if limit is not None
            else None
        )

        files: dict[str, FileImportResult] = {}
        for entity, spec in MAIN_IMPORT_SPECS.items():
            if verbose:
                print(f"[主实体] {entity}", file=sys.stderr, flush=True)
            files[entity] = _import_main(
                conn,
                entity,
                spec,
                raw_dir,
                limit=limit,
                batch_size=batch_size,
                id_sets=id_sets,
                verbose=verbose,
            )

        for name, spec in RELATION_IMPORT_SPECS.items():
            if verbose:
                print(f"[关联表] {name}", file=sys.stderr, flush=True)
            files[name] = _import_relation(
                conn,
                name,
                spec,
                raw_dir,
                limit=limit,
                batch_size=batch_size,
                id_sets=id_sets,
                verbose=verbose,
            )

        if verbose:
            print("构建索引与统计信息...", file=sys.stderr, flush=True)
        index_started = time.perf_counter()
        for statement in INDEX_SQL:
            conn.execute(statement)
        conn.execute("ANALYZE")
        conn.commit()
        index_elapsed = time.perf_counter() - index_started
        if verbose:
            print(f"索引构建完成：{index_elapsed:.1f}s", file=sys.stderr, flush=True)

        if verbose:
            print("执行完整性检查...", file=sys.stderr, flush=True)
        table_counts = _counts(conn)
        integrity = _integrity(conn)
        samples = _samples(conn)
        elapsed = time.perf_counter() - started

        _write_metadata(
            conn,
            raw_dir=raw_dir,
            mode="limit" if limit is not None else "full",
            limit=limit,
            report_path=report_path,
            files=files,
            table_counts=table_counts,
            integrity=integrity,
            elapsed_sec=elapsed,
        )
        conn.execute("PRAGMA optimize")
        conn.commit()
        conn.close()
        conn = None

        os.replace(temporary_db, db_path)
        db_size = db_path.stat().st_size
        report = {
            "generated_at": started_at.isoformat(timespec="seconds"),
            "command": command,
            "mode": "limit" if limit is not None else "full",
            "limit": limit,
            "batch_size": batch_size,
            "schema_version": SCHEMA_VERSION,
            "raw_dir": str(raw_dir),
            "db_path": str(db_path),
            "db_size_bytes": db_size,
            "db_size_gib": round(db_size / 1024 / 1024 / 1024, 3),
            "elapsed_sec": round(elapsed, 3),
            "peak_rss_mb": peak_rss_mb(),
            "index_build_sec": round(index_elapsed, 3),
            "table_counts": table_counts,
            "reconciliation": {result.table: result.as_dict() for result in files.values()},
            "files": {result.file: result.as_dict() for result in files.values()},
            "integrity": integrity,
            "samples": samples,
            "notes": [
                "未声明硬外键：原始归档存在悬空引用，使用导入后完整性查询替代；"
                "PRAGMA foreign_keys 已启用并验证。",
                "person_relations 的 person/character ID 空间重叠，"
                "按 person_type 做命名空间归属统计，不把单边未命中直接判为孤儿。",
                "limit 模式下 filtered_by_limit 表示关联端点不在限量主实体集合内，"
                "不是原始数据错误。",
            ],
        }
        _write_json(report_path, report)

        if verbose:
            print(f"数据库：{db_path}（{db_size / 1024 / 1024 / 1024:.2f} GiB）", file=sys.stderr, flush=True)
            print(f"报告：{report_path}", file=sys.stderr, flush=True)
            print(f"总耗时：{elapsed:.1f} 秒", file=sys.stderr, flush=True)

        return {
            "db_path": str(db_path),
            "report_path": str(report_path),
            "db_size_bytes": db_size,
            "elapsed_sec": elapsed,
            "table_counts": table_counts,
            "integrity": integrity,
            "reconciliation": {result.table: result.as_dict() for result in files.values()},
        }
    except Exception:
        if conn is not None:
            conn.close()
        if temporary_db.exists():
            temporary_db.unlink()
        raise
