from __future__ import annotations

import hashlib
import json

import pytest

from ingestion.audit import (
    FileStats,
    LengthStats,
    classify_episode_description,
    classify_text,
    iter_jsonl,
    parse_aliases,
    parse_infobox,
    run_audit,
)
from ingestion.text_quality import clean_episode_description


def test_parse_infobox_multiline_aliases() -> None:
    raw = (
        "{{Infobox Crt\r\n"
        "|简体中文名= 鲁路修·兰佩路基\r\n"
        "|别名={\r\n"
        "[L.L.]\r\n"
        "[勒鲁什]\r\n"
        "[英文名|Lelouch Lamperouge]\r\n"
        "[纯假名|]\r\n"
        "}\r\n"
        "|性别= 男\r\n"
        "}}"
    )
    info = parse_infobox(raw)
    assert info["简体中文名"] == "鲁路修·兰佩路基"
    assert info["性别"] == "男"
    aliases = parse_aliases(info["别名"])
    assert aliases == ["L.L.", "勒鲁什", "Lelouch Lamperouge"]


def test_parse_infobox_single_line_fallback() -> None:
    raw = "{{Infobox animanga/Novel|中文名= 第一次的亲密接触|别名={[A|Alpha]}|出版社= 某社}}"
    info = parse_infobox(raw)
    assert info["中文名"] == "第一次的亲密接触"
    assert info["出版社"] == "某社"
    assert parse_aliases(info["别名"]) == ["Alpha"]


def test_parse_infobox_empty_and_malformed() -> None:
    assert parse_infobox(None) == {}
    assert parse_infobox("") == {}
    assert parse_infobox("not a template") == {}


def test_parse_aliases_dedupes_and_splits_plain_text() -> None:
    assert parse_aliases("[A|X]\n[A|X]\n[A|Y]") == ["X", "Y"]
    assert parse_aliases("甲、乙，丙") == ["甲", "乙", "丙"]
    assert parse_aliases("[英文名|]") == []


def test_classify_text_scripts() -> None:
    assert classify_text("") == "empty"
    assert classify_text("这是一部动画") == "chinese_simplified"
    assert classify_text("這是一部動畫") == "chinese_traditional"
    assert classify_text("這部动画") == "chinese_mixed"
    assert classify_text("これはアニメです") == "japanese"
    assert classify_text("hello world") == "latin"


def test_classify_episode_description() -> None:
    assert classify_episode_description("") == "empty"
    assert (
        classify_episode_description(
            "シナリオ/大河内 一楼、コンテ/須永司、演出/三宅和男\r\nキャラ作監/石田 可奈"
        )
        == "credit_only"
    )
    assert (
        classify_episode_description(
            "シャーリーが死んだ。ルルーシュはその仇を討つ為にギアス嚮団の壊滅を決意する。"
        )
        == "narrative"
    )
    assert (
        classify_episode_description(
            "シナリオ/大河内 一楼\r\nシャーリーが死んだ。ルルーシュは復讐を決意する。"
        )
        == "mixed"
    )
    assert classify_episode_description("第1話") == "other"


def test_clean_episode_description_removes_staff_lines() -> None:
    cleaned, removed = clean_episode_description(
        "キャラ作監/小暮昌宏・しんぼたくろう（中村プロダクション）\n"
        "主人公は学園に戻り、少女と再会する。"
    )
    assert removed == 1
    assert "作監" not in cleaned
    assert "主人公は学園に戻り" in cleaned


def test_length_stats_percentiles() -> None:
    stats = LengthStats()
    stats.add(None)
    stats.add("  ")
    stats.add("ab")
    stats.add("abcd")
    stats.add("a" * 10)
    summary = stats.summary()
    assert summary["count"] == 5
    assert summary["empty"] == 2
    assert summary["non_empty"] == 3
    assert summary["min"] == 2
    assert summary["p50"] == 4
    assert summary["p90"] == 10
    assert summary["max"] == 10
    assert summary["mean"] == pytest.approx(5.3)


def test_iter_jsonl_counts_and_hash(tmp_path) -> None:
    path = tmp_path / "sample.jsonlines"
    path.write_text('{"a": 1}\n\nnot-json\n[1, 2]\n{"a": 2}\n', encoding="utf-8")
    stats = FileStats(name=path.name, path=path)
    records = list(iter_jsonl(path, stats))
    assert [record["a"] for _, record in records] == [1, 2]
    assert stats.total_lines == 5
    assert stats.valid_records == 2
    assert stats.blank_lines == 1
    assert stats.parse_errors == 1
    assert stats.schema_errors == 1
    assert stats.decode_errors == 0
    assert stats.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_iter_jsonl_limit_skips_hash(tmp_path) -> None:
    path = tmp_path / "sample.jsonlines"
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")
    stats = FileStats(name=path.name, path=path)
    records = list(iter_jsonl(path, stats, limit=2))
    assert len(records) == 2
    assert stats.total_lines == 2
    assert stats.partial is True
    assert stats.sha256 == ""


def _write_minimal_dataset(raw_dir) -> None:
    subject = {
        "id": 1,
        "type": 1,
        "name": "作品",
        "name_cn": "作品",
        "infobox": "{{Infobox animanga/TVAnime\n|中文名= 作品\n|别名={\n[英文名|Work]\n}\n}}",
        "platform": 1002,
        "summary": "简介",
        "nsfw": False,
        "tags": [{"name": "标签", "count": 1}],
        "meta_tags": [],
        "score": 7.0,
        "rank": 1,
        "date": "2020-01-01",
        "favorite": {},
        "series": False,
    }
    files = {
        "subject.jsonlines": [subject],
        "character.jsonlines": [
            {
                "id": 2,
                "role": 1,
                "name": "角色",
                "infobox": "{{Infobox Crt\n|简体中文名= 角色\n|别名={\n[罗马字|Kakaku]\n}\n}}",
                "summary": "角色简介",
                "comments": 0,
                "collects": 0,
            }
        ],
        "person.jsonlines": [
            {
                "id": 3,
                "name": "人物",
                "type": 1,
                "career": ["seiyu"],
                "infobox": "",
                "summary": "",
                "comments": 0,
                "collects": 0,
            }
        ],
        "episode.jsonlines": [
            {
                "id": 4,
                "name": "第1话",
                "name_cn": "第1话",
                "description": "ある日の物語。",
                "airdate": "2020-01-01",
                "disc": 0,
                "duration": "24m",
                "subject_id": 1,
                "sort": 1,
                "type": 0,
            }
        ],
        "subject-characters.jsonlines": [
            {"subject_id": 1, "character_id": 2, "type": 1, "order": 0}
        ],
        "subject-persons.jsonlines": [
            {"person_id": 3, "subject_id": 1, "position": 1, "appear_eps": ""}
        ],
        "subject-relations.jsonlines": [
            {"subject_id": 1, "relation_type": 1, "related_subject_id": 1, "order": 0}
        ],
        "person-characters.jsonlines": [
            {"person_id": 3, "subject_id": 1, "character_id": 2, "type": 0, "summary": ""}
        ],
        "person-relations.jsonlines": [
            {
                "person_type": "prsn",
                "person_id": 3,
                "related_person_id": 3,
                "relation_type": 1002,
                "spoiler": False,
                "ended": False,
            }
        ],
    }
    for filename, records in files.items():
        content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
        (raw_dir / filename).write_text(content, encoding="utf-8")


def test_run_audit_end_to_end(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "data-audit.md"
    _write_minimal_dataset(raw_dir)

    summary = run_audit(
        raw_dir=raw_dir,
        output_path=output,
        sample_size=10,
        command="python scripts/audit_data.py",
        verbose=False,
    )

    assert summary["entities"] == {"subject": 1, "character": 1, "person": 1, "episode": 1}
    assert summary["relations"]["subject-characters"] == 1
    assert summary["bad_lines"] == 0

    report = output.read_text(encoding="utf-8")
    assert "Bangumi Archive 数据审计报告" in report
    assert "首期索引范围建议" in report
    assert "SHA256" in report
    assert "Lelouch" not in report
    assert "字段缺失" in report
    assert "关联覆盖" in report
