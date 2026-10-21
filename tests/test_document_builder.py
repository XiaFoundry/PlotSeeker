from __future__ import annotations

import json

import pytest

from ingestion.document_builder import (
    SUBJECT_SUMMARY_MAX_CHARS,
    TOTAL_TEXT_MAX_CHARS,
    run_build,
)
from ingestion.importer import run_import


def _write_jsonl(path, records) -> None:
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    path.write_text(content + ("\n" if records else ""), encoding="utf-8")


def _subject(record_id: int, name: str, summary: str) -> dict:
    return {
        "id": record_id,
        "type": 2,
        "name": name,
        "name_cn": f"{name}中文",
        "infobox": (
            f"{{{{Infobox animanga/TVAnime\n"
            f"|中文名= {name}中文\n"
            f"|别名={{\n[英文名|{name} EN]\n}}\n"
            f"}}}}"
        ),
        "platform": 1002,
        "summary": summary,
        "nsfw": False,
        "tags": [{"name": "动画", "count": 10}, {"name": "校园", "count": 5}],
        "meta_tags": [],
        "score": 7.5,
        "score_details": {"7": 10},
        "rank": 100,
        "date": "2008-04-06",
        "favorite": {"wish": 1},
        "series": False,
    }


def _character(record_id: int, name: str, summary: str) -> dict:
    return {
        "id": record_id,
        "role": 1,
        "name": name,
        "infobox": (
            f"{{{{Infobox Crt\n"
            f"|简体中文名= {name}中文\n"
            f"|别名={{\n[罗马字|{name} Romaji]\n}}\n"
            f"}}}}"
        ),
        "summary": summary,
        "comments": 1,
        "collects": 2,
    }


def _person(record_id: int, name: str) -> dict:
    return {
        "id": record_id,
        "name": name,
        "type": 1,
        "career": ["seiyu"],
        "infobox": "",
        "summary": "",
        "comments": 0,
        "collects": 0,
    }


def _episode(record_id: int, subject_id: int, description: str, name: str | None = None) -> dict:
    return {
        "id": record_id,
        "name": name if name is not None else f"第{record_id}话",
        "name_cn": f"第{record_id}话",
        "description": description,
        "airdate": "2008-04-06",
        "disc": 0,
        "duration": "24m",
        "subject_id": subject_id,
        "sort": 1,
        "type": 0,
    }


def _write_dataset(raw_dir) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        raw_dir / "subject.jsonlines",
        [
            _subject(1, "作品A", "作品A简介。" + "长" * (SUBJECT_SUMMARY_MAX_CHARS + 200)),
            _subject(2, "作品B", "作品B简介，包含校园与时间循环。"),
            _subject(3, "作品C", ""),
        ],
    )
    _write_jsonl(
        raw_dir / "character.jsonlines",
        [
            _character(10, "角色A", "角色A的角色简介。"),
            _character(11, "角色B", "角色B的角色简介。"),
            _character(12, "角色C", ""),
            _character(13, "角色D", "角色D没有任何作品关联。"),
        ],
    )
    _write_jsonl(
        raw_dir / "person.jsonlines",
        [_person(20, "人物A"), _person(21, "人物B")],
    )
    _write_jsonl(
        raw_dir / "episode.jsonlines",
        [
            _episode(100, 1, "主角在校园里遇到了一位神秘少女。"),
            _episode(101, 1, "脚本：某人 演出：某人 作画监督：某人"),
            _episode(102, 1, "脚本：某人\n主角在校园里遇到了一位神秘少女！"),
            _episode(103, 1, ""),
            _episode(104, 999, "孤儿章节的剧情描述。"),
            _episode(105, 2, "短句。"),
        ],
    )
    _write_jsonl(
        raw_dir / "subject-characters.jsonlines",
        [
            {"subject_id": 1, "character_id": 10, "type": 1, "order": 0},
            {"subject_id": 2, "character_id": 10, "type": 1, "order": 1},
            {"subject_id": 1, "character_id": 11, "type": 1, "order": 2},
            {"subject_id": 1, "character_id": 12, "type": 1, "order": 3},
        ],
    )
    _write_jsonl(
        raw_dir / "subject-persons.jsonlines",
        [{"subject_id": 1, "person_id": 20, "position": 1, "appear_eps": ""}],
    )
    _write_jsonl(
        raw_dir / "subject-relations.jsonlines",
        [{"subject_id": 1, "relation_type": 1, "related_subject_id": 2, "order": 0}],
    )
    _write_jsonl(
        raw_dir / "person-characters.jsonlines",
        [{"person_id": 20, "subject_id": 1, "character_id": 10, "type": 0, "summary": ""}],
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
            }
        ],
    )


def _prepare_db(tmp_path):
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "plotseeker.db"
    _write_dataset(raw_dir)
    run_import(
        raw_dir=raw_dir,
        db_path=db_path,
        report_path=tmp_path / "import-report.json",
        verbose=False,
    )
    return db_path


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_id(documents):
    return {document["doc_id"]: document for document in documents}


def test_full_document_build(tmp_path) -> None:
    db_path = _prepare_db(tmp_path)
    output_dir = tmp_path / "documents"

    summary = run_build(
        db_path=db_path,
        output_dir=output_dir,
        verbose=False,
        command="python scripts/build_documents.py",
    )

    counts = summary["counts"]
    assert counts["subject"]["source_total"] == 3
    assert counts["subject"]["written"] == 2
    assert counts["subject"]["filtered"] == {"empty_summary": 1}
    assert counts["subject"]["reconciled"] is True

    assert counts["character"]["source_total"] == 4
    assert counts["character"]["written"] == 2
    assert counts["character"]["filtered"] == {"empty_summary": 1, "no_subject": 1}
    assert counts["character"]["reconciled"] is True

    assert counts["episode"]["source_total"] == 6
    assert counts["episode"]["written"] == 2
    assert counts["episode"]["filtered"] == {
        "credit_only": 1,
        "empty_description": 1,
        "orphan_subject": 1,
        "too_short_or_unclear": 1,
    }
    assert counts["episode"]["reconciled"] is True

    subjects = _by_id(_read_jsonl(output_dir / "subjects.jsonl"))
    characters = _by_id(_read_jsonl(output_dir / "characters.jsonl"))
    episodes = _by_id(_read_jsonl(output_dir / "episodes.jsonl"))

    assert set(subjects) == {"subject:1", "subject:2"}
    assert set(characters) == {"character:10", "character:11"}
    assert set(episodes) == {"episode:100", "episode:102"}

    subject = subjects["subject:1"]
    assert subject["source"] == "subject.jsonlines"
    assert subject["subject_id"] == 1
    assert "作品A" in subject["text"]
    assert "作品A EN" in subject["text"]
    assert subject["metadata"]["aliases"] == ["作品A EN"]
    assert "summary" in subject["metadata"]["truncated_fields"]
    assert len(subject["text"]) <= TOTAL_TEXT_MAX_CHARS["subject"]
    assert subject["metadata"]["tags"] == ["动画", "校园"]

    character = characters["character:10"]
    assert character["subject_id"] == 1
    assert character["metadata"]["subject_ids"] == [1, 2]
    assert character["metadata"]["subject_count"] == 2
    assert "作品A" in character["text"]
    assert "作品B" in character["text"]
    assert "角色A" in character["text"]

    episode = episodes["episode:102"]
    assert episode["subject_id"] == 1
    assert episode["metadata"]["quality_category"] == "mixed"
    assert episode["metadata"]["removed_credit_lines"] == 1
    assert "神秘少女" in episode["text"]
    assert "脚本" not in episode["text"]

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rules_version"] == "p3-v1"
    assert manifest["traceability"]["sample_size"] == manifest["traceability"]["verified"] == 6
    assert manifest["traceability"]["failures"] == []
    assert manifest["files"]["subject"]["lines"] == 2
    assert len(manifest["files"]["subject"]["sha256"]) == 64


def test_rebuild_protection_and_determinism(tmp_path) -> None:
    db_path = _prepare_db(tmp_path)
    output_dir = tmp_path / "documents"

    first = run_build(db_path=db_path, output_dir=output_dir, verbose=False)
    first_hashes = {
        name: info["sha256"] for name, info in first["files"].items()
    }

    with pytest.raises(FileExistsError):
        run_build(db_path=db_path, output_dir=output_dir, verbose=False)

    second = run_build(
        db_path=db_path, output_dir=output_dir, overwrite=True, verbose=False
    )
    second_hashes = {
        name: info["sha256"] for name, info in second["files"].items()
    }
    assert first_hashes == second_hashes
    assert first["counts"] == second["counts"]


def test_limit_mode(tmp_path) -> None:
    db_path = _prepare_db(tmp_path)
    output_dir = tmp_path / "documents-limit"

    summary = run_build(
        db_path=db_path, output_dir=output_dir, limit=1, verbose=False
    )

    assert summary["counts"]["subject"]["written"] == 1
    assert summary["counts"]["character"]["written"] == 1
    assert summary["counts"]["episode"]["written"] == 1
    assert summary["counts"]["subject"]["reconciled"] is None
    assert summary["files"]["subject"]["lines"] == 1
    assert summary["traceability"]["sample_size"] == 3

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "limit"
    assert manifest["limit"] == 1
