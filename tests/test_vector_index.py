from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from embeddings.chroma_store import create_client
from embeddings.fake import DeterministicFakeEmbedder
from embeddings.indexer import (
    COLLECTION_NAMES,
    IndexAlreadyComplete,
    IndexConflict,
    IndexInterrupted,
    SourceChanged,
    _derive_seed,
    _sample_documents,
    run_build_index,
)


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


def _write_dataset(root: Path, *, ids: int = 3) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    records = {
        "subject": [
            _document(
                "subject",
                index,
                f"作品名称：作品{index}\n标签：校园、时间循环、悬疑\n简介："
                f"第{index}个作品的剧情简介，主角在校园中寻找真相。",
                metadata={
                    "name": f"作品{index}",
                    "aliases": ["别名A", "别名B"],
                    "tags": ["校园", "时间循环"],
                    "score": 7.5 + index / 10,
                    "truncated_fields": [],
                },
            )
            for index in range(1, ids + 1)
        ],
        "character": [
            _document(
                "character",
                index,
                f"角色名称：角色{index}\n所属作品：作品1\n角色简介："
                f"第{index}个角色能够感知时间循环。",
                subject_id=1,
                metadata={
                    "name": f"角色{index}",
                    "aliases": [f"角色别名{index}"],
                    "subject_ids": [1, 2],
                },
            )
            for index in range(1, ids + 1)
        ],
        "episode": [
            _document(
                "episode",
                index,
                f"所属作品：作品1\n章节名称：第{index}话\n剧情："
                f"主角在第{index}话发现妹妹用脚抵住门。",
                subject_id=1,
                metadata={
                    "name": f"第{index}话",
                    "quality_category": "narrative",
                    "sort": float(index),
                },
            )
            for index in range(1, ids + 1)
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

    manifest = {
        "command": "test",
        "mode": "full",
        "files": files,
        "counts": {},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def _build(
    manifest_path: Path,
    store_dir: Path,
    *,
    embedder=None,
    sample_sizes=None,
    batch_size: int = 2,
    rebuild: bool = False,
    stop_after_batches: int | None = None,
    verify_source: bool = True,
) -> dict:
    return run_build_index(
        manifest_path=manifest_path,
        store_dir=store_dir,
        checkpoint_path=store_dir / "checkpoint.json",
        report_path=store_dir / "report.json",
        embedder=embedder or DeterministicFakeEmbedder(dimension=16),
        sample_sizes=sample_sizes or {"subject": 3, "character": 3, "episode": 3},
        seed=7,
        batch_size=batch_size,
        rebuild=rebuild,
        verify_source=verify_source,
        sanity_queries=3,
        stop_after_batches=stop_after_batches,
        command="test",
        verbose=False,
    )


def test_build_reopen_and_metadata_mapping(tmp_path):
    manifest_path = _write_dataset(tmp_path / "docs")
    store_dir = tmp_path / "store"
    report = _build(manifest_path, store_dir)

    assert report["index"]["reopened"] is True
    assert report["index"]["counts_verified"] is True
    for entity_type, name in COLLECTION_NAMES.items():
        assert report["documents"][entity_type]["reopened_count"] == 3
        assert report["documents"][entity_type]["verified_ids"] == {
            "expected": 3,
            "verified": 3,
        }
        assert name in report["index"]["collections"]

    assert report["sanity"]["query_count"] == 3
    assert report["sanity"]["hit_at_5"] == 3
    assert (store_dir / "checkpoint.json").is_file()
    assert json.loads((store_dir / "checkpoint.json").read_text("utf-8"))["state"] == (
        "complete"
    )

    client = create_client(store_dir)
    collection = client.get_collection(COLLECTION_NAMES["character"])
    result = collection.get(
        ids=["character:1"], include=["metadatas", "documents"]
    )
    metadata = result["metadatas"][0]
    assert metadata["entity_id"] == 1
    assert metadata["subject_id"] == 1
    assert metadata["aliases"] == json.dumps(["角色别名1"], ensure_ascii=False)
    assert metadata["subject_ids"] == json.dumps([1, 2], ensure_ascii=False)
    assert isinstance(metadata["aliases"], str)
    assert result["documents"][0].startswith("角色名称：角色1")


def test_resume_after_interruption(tmp_path):
    manifest_path = _write_dataset(tmp_path / "docs")
    store_dir = tmp_path / "store"

    with pytest.raises(IndexInterrupted):
        _build(manifest_path, store_dir, stop_after_batches=2)
    from embeddings.chroma_store import clear_client_cache

    clear_client_cache()

    checkpoint = json.loads((store_dir / "checkpoint.json").read_text("utf-8"))
    processed = sum(
        len(entry["processed_ids"]) for entry in checkpoint["collections"].values()
    )
    assert checkpoint["state"] == "in_progress"
    assert processed == 3

    report = _build(manifest_path, store_dir)
    assert report["checkpoint"]["resumed"] is True
    assert report["checkpoint"]["new_batches_this_run"] == 4
    for entity_type in COLLECTION_NAMES:
        assert report["documents"][entity_type]["reopened_count"] == 3
    final_checkpoint = json.loads(
        (store_dir / "checkpoint.json").read_text("utf-8")
    )
    assert final_checkpoint["state"] == "complete"


def test_conflict_model_change_and_rebuild(tmp_path):
    manifest_path = _write_dataset(tmp_path / "docs")
    store_dir = tmp_path / "store"
    first = _build(manifest_path, store_dir, embedder=DeterministicFakeEmbedder(16))

    with pytest.raises(IndexAlreadyComplete):
        _build(manifest_path, store_dir, embedder=DeterministicFakeEmbedder(16))

    with pytest.raises(IndexConflict):
        _build(manifest_path, store_dir, embedder=DeterministicFakeEmbedder(32))

    rebuilt = _build(
        manifest_path,
        store_dir,
        embedder=DeterministicFakeEmbedder(32),
        rebuild=True,
    )
    assert rebuilt["rebuild"]["performed"] is True
    assert rebuilt["rebuild"]["old_index"]["config_id"] == first["config_id"]
    assert rebuilt["embedding"]["dimension"] == 32


def test_source_hash_mismatch_is_rejected(tmp_path):
    manifest_path = _write_dataset(tmp_path / "docs")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["files"]["subject"]["sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(SourceChanged):
        _build(manifest_path, tmp_path / "store")


def test_sampling_is_deterministic(tmp_path):
    manifest_path = _write_dataset(tmp_path / "docs", ids=8)
    for entity_type in ("subject", "character", "episode"):
        path = tmp_path / "docs" / f"{entity_type}.jsonl"
        seed = _derive_seed(7, entity_type)
        first, total_a = _sample_documents(path, 4, seed)
        second, total_b = _sample_documents(path, 4, seed)
        assert total_a == total_b == 8
        assert [item["doc_id"] for item in first] == [
            item["doc_id"] for item in second
        ]
        assert len(first) == 4
