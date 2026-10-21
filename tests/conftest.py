from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from embeddings.chroma_store import clear_client_cache
from embeddings.fake import DeterministicFakeEmbedder
from embeddings.indexer import run_build_index
from retrieval.keyword_index import build_keyword_index


@pytest.fixture(autouse=True)
def _isolate_chroma_clients():
    clear_client_cache()
    yield
    clear_client_cache()


def _document(
    entity_type: str,
    entity_id: int,
    text: str,
    *,
    subject_id: int | None = None,
    metadata: dict | None = None,
) -> dict:
    if subject_id is None:
        subject_id = entity_id if entity_type == "subject" else 1
    return {
        "doc_id": f"{entity_type}:{entity_id}",
        "entity_type": entity_type,
        "entity_id": entity_id,
        "subject_id": subject_id,
        "source": f"{entity_type}.jsonlines",
        "text": text,
        "metadata": metadata or {},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    records = {
        "subject": [
            _document(
                "subject",
                1,
                "作品名称：时间循环学园\n中文名称：时间循环学园\n标签：校园、时间循环\n"
                "简介：主角在校园中不断经历时间循环，并试图找到真相。",
                metadata={"name": "时间循环学园", "name_cn": "时间循环学园"},
            ),
            _document(
                "subject",
                2,
                "作品名称：魔王转生记\n中文名称：魔王转生记\n标签：魔王、学校\n"
                "简介：失去记忆的魔王在现代学校生活。",
                metadata={"name": "魔王转生记", "name_cn": "魔王转生记"},
            ),
            _document(
                "subject",
                3,
                "作品名称：白色生物\n中文名称：白色生物\n标签：奇幻\n"
                "简介：一个会说话的白色生物。",
                metadata={"name": "白色生物", "name_cn": "白色生物"},
            ),
        ],
        "character": [
            _document(
                "character",
                1,
                "角色名称：小焰\n所属作品：时间循环学园\n"
                "角色简介：她能感知时间循环，并试图阻止妹妹打开门。",
                subject_id=1,
                metadata={
                    "name": "小焰",
                    "subject_ids": [1, 2],
                    "subject_names": ["时间循环学园", "魔王转生记"],
                },
            ),
            _document(
                "character",
                2,
                "角色名称：魔王\n所属作品：魔王转生记\n角色简介：失去记忆的魔王。",
                subject_id=2,
                metadata={"name": "魔王", "subject_ids": [2]},
            ),
            _document(
                "character",
                3,
                "角色名称：白色\n所属作品：白色生物\n角色简介：会说话的白色生物。",
                subject_id=3,
                metadata={"name": "白色", "subject_ids": [3]},
            ),
        ],
        "episode": [
            _document(
                "episode",
                1,
                "所属作品：时间循环学园\n章节名称：门\n"
                "剧情：妹妹用脚抵住门，不让人进来。",
                subject_id=1,
                metadata={"name": "门", "quality_category": "narrative"},
            ),
            _document(
                "episode",
                2,
                "所属作品：魔王转生记\n章节名称：转生\n"
                "剧情：魔王在教室醒来，发现自己失去了记忆。",
                subject_id=2,
                metadata={"name": "转生", "quality_category": "narrative"},
            ),
            _document(
                "episode",
                3,
                "所属作品：白色生物\n章节名称：相遇\n"
                "剧情：主角遇见会说话的白色生物。",
                subject_id=3,
                metadata={"name": "相遇", "quality_category": "narrative"},
            ),
        ],
    }

    files = {}
    for entity_type, items in records.items():
        path = root / f"{entity_type}.jsonl"
        _write_jsonl(path, items)
        files[entity_type] = {
            "path": str(path),
            "lines": len(items),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest = {"command": "test", "mode": "full", "files": files, "counts": {}}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def _write_subject_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE subject(id INTEGER PRIMARY KEY, name TEXT, name_cn TEXT, "
        "type INTEGER, platform INTEGER, date TEXT, score REAL, nsfw INTEGER)"
    )
    connection.execute(
        "CREATE TABLE subject_relations(subject_id INTEGER, related_subject_id INTEGER, "
        "relation_type INTEGER, ord INTEGER)"
    )
    connection.executemany(
        "INSERT INTO subject_relations VALUES (?, ?, ?, ?)",
        [(1, 2, 1, 0), (2, 1, 2, 0), (1, 3, 3, 1)],
    )
    connection.executemany(
        "INSERT INTO subject VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "時間循環学園", "时间循环学园", 2, 1002, "2020-01-01", 8.0, 0),
            (2, "魔王転生記", "魔王转生记", 2, 1002, "2021-01-01", 7.5, 0),
            (3, "白い生物", "白色生物", 2, 1002, "2022-01-01", 7.0, 0),
        ],
    )
    connection.commit()
    connection.close()


@pytest.fixture
def retrieval_env(tmp_path):
    manifest_path = _write_dataset(tmp_path / "docs")
    store_dir = tmp_path / "vector_store"
    checkpoint_path = store_dir / "p4_checkpoint.json"
    fake = DeterministicFakeEmbedder(dimension=16)
    run_build_index(
        manifest_path=manifest_path,
        store_dir=store_dir,
        checkpoint_path=checkpoint_path,
        report_path=store_dir / "p4_report.json",
        embedder=fake,
        sample_sizes={"subject": 3, "character": 3, "episode": 3},
        seed=7,
        batch_size=2,
        sanity_queries=0,
        command="test",
        verbose=False,
    )
    keyword_db = tmp_path / "keyword.db"
    keyword_report = tmp_path / "keyword_report.json"
    build_keyword_index(
        manifest_path=manifest_path,
        p4_checkpoint_path=checkpoint_path,
        output_path=keyword_db,
        report_path=keyword_report,
        rebuild=False,
        verify_source=True,
        verbose=False,
    )
    sqlite_path = tmp_path / "subjects.db"
    _write_subject_db(sqlite_path)
    return {
        "manifest": manifest_path,
        "store_dir": store_dir,
        "checkpoint": checkpoint_path,
        "keyword_db": keyword_db,
        "keyword_report": keyword_report,
        "sqlite": sqlite_path,
        "fake_dimension": 16,
    }
