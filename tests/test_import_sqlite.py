from __future__ import annotations

import json
import sqlite3

import pytest

from ingestion.importer import SCHEMA_VERSION, run_import


def _write_jsonl(path, records) -> None:
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    path.write_text(content + ("\n" if records else ""), encoding="utf-8")


def _subject(record_id: int, name: str) -> dict:
    return {
        "id": record_id,
        "type": 2,
        "name": name,
        "name_cn": f"{name}中文",
        "infobox": f"{{{{Infobox animanga/TVAnime\n|中文名= {name}中文\n|别名={{\n[英文名|{name} EN]\n}}\n}}}}",
        "platform": 1002,
        "summary": f"{name} 的简介。",
        "nsfw": False,
        "tags": [{"name": "动画", "count": 10}],
        "meta_tags": [],
        "score": 7.5,
        "score_details": {"7": 10},
        "rank": 100,
        "date": "2008-04-06",
        "favorite": {"wish": 1, "done": 2},
        "series": False,
    }


def _character(record_id: int, name: str) -> dict:
    return {
        "id": record_id,
        "role": 1,
        "name": name,
        "infobox": f"{{{{Infobox Crt\n|简体中文名= {name}\n}}}}",
        "summary": f"{name} 的角色简介。",
        "comments": 3,
        "collects": 4,
    }


def _person(record_id: int, name: str) -> dict:
    return {
        "id": record_id,
        "name": name,
        "type": 1,
        "career": ["seiyu"],
        "infobox": "",
        "summary": f"{name} 的人物简介。",
        "comments": 5,
        "collects": 6,
    }


def _episode(record_id: int, subject_id: int) -> dict:
    return {
        "id": record_id,
        "name": f"第{record_id}话",
        "name_cn": f"第{record_id}话",
        "description": "主角在校园里遇到了一位神秘的少女。",
        "airdate": "2008-04-06",
        "disc": 0,
        "duration": "24m",
        "subject_id": subject_id,
        "sort": 1,
        "type": 0,
    }


def _write_dataset(
    raw_dir,
    *,
    duplicate_relation: bool = False,
    orphan_relation: bool = False,
    malformed_line: bool = False,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(raw_dir / "subject.jsonlines", [_subject(1, "作品A"), _subject(2, "作品B")])
    _write_jsonl(raw_dir / "character.jsonlines", [_character(10, "角色A"), _character(11, "角色B")])
    _write_jsonl(raw_dir / "person.jsonlines", [_person(20, "人物A"), _person(21, "人物B")])
    _write_jsonl(raw_dir / "episode.jsonlines", [_episode(100, 1), _episode(101, 2)])

    subject_characters = [
        {"subject_id": 1, "character_id": 10, "type": 1, "order": 0},
        {"subject_id": 2, "character_id": 11, "type": 1, "order": 0},
    ]
    if duplicate_relation:
        subject_characters.append({"subject_id": 1, "character_id": 10, "type": 2, "order": 1})
    if orphan_relation:
        subject_characters.append({"subject_id": 999, "character_id": 10, "type": 1, "order": 2})
    _write_jsonl(raw_dir / "subject-characters.jsonlines", subject_characters)

    _write_jsonl(
        raw_dir / "subject-persons.jsonlines",
        [
            {"subject_id": 1, "person_id": 20, "position": 1, "appear_eps": ""},
            {"subject_id": 2, "person_id": 21, "position": 1, "appear_eps": ""},
        ],
    )
    _write_jsonl(
        raw_dir / "subject-relations.jsonlines",
        [
            {"subject_id": 1, "relation_type": 1, "related_subject_id": 2, "order": 0},
            {"subject_id": 2, "relation_type": 1, "related_subject_id": 1, "order": 0},
        ],
    )
    _write_jsonl(
        raw_dir / "person-characters.jsonlines",
        [
            {"person_id": 20, "subject_id": 1, "character_id": 10, "type": 0, "summary": ""},
            {"person_id": 21, "subject_id": 2, "character_id": 11, "type": 0, "summary": ""},
        ],
    )
    _write_jsonl(
        raw_dir / "person-relations.jsonlines",
        [
            {
                "person_type": "prsn",
                "person_id": 20,
                "related_person_id": 21,
                "relation_type": 1002,
                "spoiler": False,
                "ended": False,
            },
            {
                "person_type": "crt",
                "person_id": 10,
                "related_person_id": 11,
                "relation_type": 2006,
                "spoiler": False,
                "ended": False,
            },
        ],
    )

    if malformed_line:
        with (raw_dir / "subject.jsonlines").open("a", encoding="utf-8") as handle:
            handle.write("not-json\n")


def _write_limit_dataset(raw_dir) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(raw_dir / "subject.jsonlines", [_subject(1, "作品A"), _subject(2, "作品B")])
    _write_jsonl(raw_dir / "character.jsonlines", [_character(10, "角色A"), _character(11, "角色B")])
    _write_jsonl(raw_dir / "person.jsonlines", [_person(20, "人物A"), _person(21, "人物B")])
    _write_jsonl(raw_dir / "episode.jsonlines", [_episode(100, 1), _episode(101, 999)])

    _write_jsonl(
        raw_dir / "subject-characters.jsonlines",
        [
            {"subject_id": 999, "character_id": 10, "type": 1, "order": 0},
            {"subject_id": 1, "character_id": 10, "type": 1, "order": 1},
        ],
    )
    for filename in (
        "subject-persons.jsonlines",
        "subject-relations.jsonlines",
        "person-characters.jsonlines",
        "person-relations.jsonlines",
    ):
        _write_jsonl(raw_dir / filename, [])


def _query_one(db_path, sql: str, params=()):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(sql, params).fetchone()


def test_full_import_and_report(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "plotseeker.db"
    report_path = tmp_path / "import-report.json"
    _write_dataset(raw_dir, duplicate_relation=True, orphan_relation=True)

    summary = run_import(
        raw_dir=raw_dir,
        db_path=db_path,
        report_path=report_path,
        verbose=False,
        command="python scripts/import_sqlite.py",
    )

    assert db_path.exists()
    assert report_path.exists()
    assert summary["table_counts"] == {
        "subject": 2,
        "character": 2,
        "person": 2,
        "episode": 2,
        "subject_characters": 3,
        "subject_persons": 2,
        "subject_relations": 2,
        "person_characters": 2,
        "person_relations": 2,
    }

    reconciliation = summary["reconciliation"]["subject_characters"]
    assert reconciliation["valid_records"] == 4
    assert reconciliation["inserted"] == 3
    assert reconciliation["ignored"] == 1
    assert reconciliation["reconciled"] is True

    integrity = summary["integrity"]
    assert integrity["pragma_integrity_check"] == "ok"
    assert integrity["pragma_foreign_keys"] == 1
    assert integrity["foreign_key_violations"] == 0
    assert integrity["orphans"]["subject_characters.subject_id"] == 1
    assert integrity["orphans"]["episode.subject_id"] == 0
    assert integrity["primary_key_duplicates"]["subject"] == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["mode"] == "full"
    assert report["table_counts"]["subject"] == 2
    assert report["samples"]["character_to_subject"]
    assert report["samples"]["episode_to_subject"]
    assert report["samples"]["subject_relations"]
    assert report["samples"]["person_character"]
    assert "未声明硬外键" in report["notes"][0]

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT id, name, tags_json, source_file FROM subject WHERE id = 1").fetchone()
        assert row[0] == 1
        assert row[1] == "作品A"
        assert json.loads(row[2]) == [{"name": "动画", "count": 10}]
        assert row[3] == "subject.jsonlines"
        metadata = dict(conn.execute("SELECT key, value FROM import_metadata").fetchall())
        assert json.loads(metadata["schema_version"]) == SCHEMA_VERSION
        assert json.loads(metadata["table_counts"])["subject"] == 2


def test_rebuild_protection_and_idempotence(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "plotseeker.db"
    report_path = tmp_path / "import-report.json"
    _write_dataset(raw_dir)

    first = run_import(
        raw_dir=raw_dir,
        db_path=db_path,
        report_path=report_path,
        verbose=False,
    )

    with pytest.raises(FileExistsError):
        run_import(
            raw_dir=raw_dir,
            db_path=db_path,
            report_path=report_path,
            verbose=False,
        )

    second = run_import(
        raw_dir=raw_dir,
        db_path=db_path,
        report_path=report_path,
        overwrite=True,
        verbose=False,
    )
    assert first["table_counts"] == second["table_counts"]
    assert second["table_counts"]["subject"] == 2


def test_limit_mode_filters_relations_without_treating_them_as_orphans(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "plotseeker-smoke.db"
    report_path = tmp_path / "import-report-smoke.json"
    _write_limit_dataset(raw_dir)

    summary = run_import(
        raw_dir=raw_dir,
        db_path=db_path,
        report_path=report_path,
        limit=2,
        verbose=False,
    )

    assert summary["table_counts"]["subject"] == 2
    assert summary["table_counts"]["episode"] == 1
    assert summary["table_counts"]["subject_characters"] == 1

    episode_result = summary["reconciliation"]["episode"]
    assert episode_result["filtered_by_limit"] == 1
    assert episode_result["reconciled"] is True

    result = summary["reconciliation"]["subject_characters"]
    assert result["valid_records"] == 2
    assert result["filtered_by_limit"] == 1
    assert result["inserted"] == 1
    assert result["reconciled"] is True
    assert summary["integrity"]["orphans"]["subject_characters.subject_id"] == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "limit"
    assert report["limit"] == 2
    assert report["files"]["subject-characters.jsonlines"]["filtered_by_limit"] == 1
    assert report["files"]["subject.jsonlines"]["partial"] is True
    assert report["files"]["subject.jsonlines"]["sha256"] == ""


def test_malformed_json_line_is_reported_not_fatal(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "plotseeker.db"
    report_path = tmp_path / "import-report.json"
    _write_dataset(raw_dir, malformed_line=True)

    summary = run_import(
        raw_dir=raw_dir,
        db_path=db_path,
        report_path=report_path,
        verbose=False,
    )

    result = summary["reconciliation"]["subject"]
    assert result["valid_records"] == 2
    assert result["bad_lines"] == 1
    assert result["error_examples"]
    assert summary["table_counts"]["subject"] == 2
    assert _query_one(db_path, "SELECT COUNT(*) FROM subject")[0] == 2
