"""P4 vector index construction, checkpointing, verification and reporting."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from embeddings.base import Embedder
from embeddings.chroma_store import (
    IndexConflict,
    cleanup_orphan_segments,
    clear_client_cache,
    create_client,
    delete_collections,
    existing_ids,
    get_or_create_collection,
    index_size_bytes,
    query_collection,
    upsert_batch,
)
from ingestion.resources import peak_rss_mb

FORMAT_VERSION = "p4-v1"
CHECKPOINT_VERSION = "p4-checkpoint-v1"
REPORT_VERSION = "p4-report-v1"

ENTITY_TYPES = ("subject", "character", "episode")
COLLECTION_NAMES = {
    "subject": "p4_subject_docs",
    "character": "p4_character_docs",
    "episode": "p4_episode_docs",
}
DEFAULT_SAMPLE_SIZES = {"subject": 4000, "character": 3000, "episode": 3000}
DEFAULT_BATCH_SIZE = 16
DEFAULT_SANITY_QUERIES = 30
HNSW_CONFIGURATION = {
    "space": "cosine",
    "ef_construction": 200,
    "max_neighbors": 64,
    "ef_search": 200,
}


class IndexInterrupted(RuntimeError):
    """Raised by the test/debug hook after a completed batch."""


class IndexAlreadyComplete(RuntimeError):
    """Raised when a complete index exists and --rebuild was not requested."""


class SourceChanged(RuntimeError):
    """Raised when the source documents no longer match their manifest."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile / 100.0))
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def _derive_seed(seed: int, entity_type: str) -> int:
    material = f"{seed}:{entity_type}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _load_source_manifest(
    manifest_path: Path, *, verify_hashes: bool
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"文档清单不存在：{manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("文档清单缺少 files 字段")

    sources: dict[str, dict[str, Any]] = {}
    for entity_type in ENTITY_TYPES:
        entry = files.get(entity_type)
        if not isinstance(entry, dict):
            raise ValueError(f"文档清单缺少 {entity_type} 文件信息")
        path = _resolve_manifest_path(manifest_path, entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"文档文件不存在：{path}")
        expected_sha = entry.get("sha256")
        actual_sha = _sha256_file(path) if verify_hashes else None
        if verify_hashes and expected_sha and actual_sha != expected_sha:
            raise SourceChanged(
                f"{entity_type} 文档哈希与清单不一致："
                f"expected={expected_sha} actual={actual_sha}"
            )
        sources[entity_type] = {
            "path": str(path.resolve()),
            "expected_lines": int(entry.get("lines", 0)),
            "sha256": expected_sha,
            "verified_sha256": actual_sha,
        }

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return manifest, sources, manifest_sha256


def _sample_documents(
    path: Path, sample_size: int, seed: int
) -> tuple[list[dict[str, Any]], int]:
    if sample_size < 0:
        raise ValueError("sample size must be non-negative")
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} JSON 解析失败：{exc}") from exc
            for field in ("doc_id", "entity_type", "entity_id", "text"):
                if field not in document:
                    raise ValueError(f"{path}:{line_number} 缺少字段 {field}")
            if not isinstance(document["text"], str) or not document["text"].strip():
                raise ValueError(f"{path}:{line_number} 文本为空")
            seen += 1
            if sample_size == 0:
                continue
            if len(reservoir) < sample_size:
                reservoir.append(document)
            else:
                index = rng.randrange(seen)
                if index < sample_size:
                    reservoir[index] = document
    reservoir.sort(key=lambda item: str(item["doc_id"]))
    return reservoir, seen


def _select_samples(
    sources: dict[str, dict[str, Any]],
    sample_sizes: dict[str, int],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    samples: dict[str, list[dict[str, Any]]] = {}
    actual_sizes: dict[str, int] = {}
    for index, entity_type in enumerate(ENTITY_TYPES):
        source = sources[entity_type]
        path = Path(source["path"])
        documents, seen = _sample_documents(
            path,
            int(sample_sizes[entity_type]),
            _derive_seed(seed, entity_type),
        )
        if source["expected_lines"] and seen != source["expected_lines"]:
            raise SourceChanged(
                f"{entity_type} 行数与清单不一致："
                f"expected={source['expected_lines']} actual={seen}"
            )
        ids = [str(document["doc_id"]) for document in documents]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{entity_type} 抽样出现重复 doc_id")
        samples[entity_type] = documents
        actual_sizes[entity_type] = len(documents)
    return samples, actual_sizes


def _sample_fingerprint(samples: dict[str, list[dict[str, Any]]]) -> str:
    digest = hashlib.sha256()
    for entity_type in ENTITY_TYPES:
        for document in samples[entity_type]:
            digest.update(str(document["doc_id"]).encode("utf-8"))
            digest.update(b"\n")
        digest.update(b"---\n")
    return digest.hexdigest()


def _identity_embedder_info(embedder: Embedder) -> dict[str, Any]:
    info = embedder.info()
    return {
        "model_id": info.get("model_id"),
        "revision": info.get("revision"),
        "dimension": info.get("dimension"),
        "max_length": info.get("max_length"),
        "pooling": info.get("pooling"),
        "normalize": info.get("normalize"),
    }


def _build_identity(
    *,
    embedder: Embedder,
    sources: dict[str, dict[str, Any]],
    manifest_sha256: str,
    requested_sizes: dict[str, int],
    actual_sizes: dict[str, int],
    fingerprint: str,
    seed: int,
    batch_size: int,
    hnsw_configuration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "manifest_sha256": manifest_sha256,
        "sources": {
            entity_type: {
                "sha256": sources[entity_type]["sha256"],
                "lines": sources[entity_type]["expected_lines"],
            }
            for entity_type in ENTITY_TYPES
        },
        "embedder": _identity_embedder_info(embedder),
        "requested_sample_sizes": requested_sizes,
        "actual_sample_sizes": actual_sizes,
        "sample_fingerprint": fingerprint,
        "seed": seed,
        "batch_size": batch_size,
        "collections": dict(COLLECTION_NAMES),
        "index_configuration": {"hnsw": dict(hnsw_configuration)},
    }


def _config_id(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _new_checkpoint(
    identity: dict[str, Any],
    config_id: str,
    samples: dict[str, list[dict[str, Any]]],
    batch_size: int,
) -> dict[str, Any]:
    created = _now()
    collections: dict[str, Any] = {}
    for entity_type in ENTITY_TYPES:
        name = COLLECTION_NAMES[entity_type]
        collections[name] = {
            "entity_type": entity_type,
            "total": len(samples[entity_type]),
            "batch_size": batch_size,
            "completed_batches": 0,
            "processed_ids": [],
        }
    return {
        "format": CHECKPOINT_VERSION,
        "config_id": config_id,
        "identity": identity,
        "state": "in_progress",
        "created_at": created,
        "updated_at": created,
        "collections": collections,
    }


def _summarize_old_index(checkpoint: dict[str, Any]) -> dict[str, Any]:
    identity = checkpoint.get("identity") or {}
    counts: dict[str, int] = {}
    for name, entry in (checkpoint.get("collections") or {}).items():
        counts[name] = len(entry.get("processed_ids") or [])
    return {
        "config_id": checkpoint.get("config_id"),
        "state": checkpoint.get("state"),
        "updated_at": checkpoint.get("updated_at"),
        "embedder": identity.get("embedder"),
        "counts": counts,
    }


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    identity: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]],
) -> None:
    if checkpoint.get("format") != CHECKPOINT_VERSION:
        raise IndexConflict("checkpoint 格式不支持")
    if checkpoint.get("identity") != identity:
        raise IndexConflict("checkpoint 身份信息与当前构建参数不一致")
    collections = checkpoint.get("collections")
    if not isinstance(collections, dict):
        raise IndexConflict("checkpoint 缺少 collections")
    for entity_type in ENTITY_TYPES:
        name = COLLECTION_NAMES[entity_type]
        entry = collections.get(name)
        if not isinstance(entry, dict):
            raise IndexConflict(f"checkpoint 缺少 {name}")
        total = int(entry.get("total", -1))
        if total != len(samples[entity_type]):
            raise IndexConflict(f"{name} 抽样总数与 checkpoint 不一致")
        processed_ids = entry.get("processed_ids")
        if not isinstance(processed_ids, list):
            raise IndexConflict(f"{name} 缺少 processed_ids")
        batch_size = max(1, int(entry.get("batch_size", 1)))
        completed_batches = int(entry.get("completed_batches", 0))
        expected_processed = min(completed_batches * batch_size, total)
        if len(processed_ids) != expected_processed:
            raise IndexConflict(
                f"{name} processed_ids 与批次数不一致："
                f"ids={len(processed_ids)} expected={expected_processed}"
            )


def _verify_resume_ids(client, checkpoint: dict[str, Any]) -> None:
    for name, entry in checkpoint["collections"].items():
        processed_ids = [str(value) for value in entry.get("processed_ids", [])]
        if not processed_ids:
            continue
        try:
            collection = client.get_collection(name)
        except Exception as exc:  # noqa: BLE001 - collection may be missing
            raise IndexConflict(f"{name} 在索引中不存在，无法恢复") from exc
        found = existing_ids(collection, processed_ids)
        missing = [doc_id for doc_id in processed_ids if doc_id not in found]
        if missing:
            raise IndexConflict(
                f"{name} 有 {len(missing)} 条 checkpoint 记录在索引中缺失，"
                "请使用 --rebuild 重建"
            )


def _build_sanity_queries(
    samples: dict[str, list[dict[str, Any]]], limit: int
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    per_type = max(1, math.ceil(limit / len(ENTITY_TYPES)))
    queries: list[dict[str, Any]] = []
    for entity_type in ENTITY_TYPES:
        documents = samples[entity_type]
        if not documents:
            continue
        count = min(per_type, len(documents))
        if count == 1:
            indices = [0]
        else:
            indices = sorted(
                {
                    int(round(index * (len(documents) - 1) / (count - 1)))
                    for index in range(count)
                }
            )
        for index in indices:
            document = documents[index]
            query = document["text"][:200].strip()
            if not query:
                continue
            queries.append(
                {
                    "entity_type": entity_type,
                    "collection": COLLECTION_NAMES[entity_type],
                    "expected_doc_id": document["doc_id"],
                    "query": query,
                    "expected_entity_id": document["entity_id"],
                }
            )
    return queries[:limit]


def _run_sanity_checks(
    client,
    embedder: Embedder,
    queries: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    hit_at_1 = 0
    hit_at_5 = 0
    failures: list[dict[str, Any]] = []
    for query in queries:
        started = time.perf_counter()
        embedding = embedder.encode_queries([query["query"]])[0]
        collection = client.get_collection(query["collection"])
        rows = query_collection(collection, embedding, top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(latency_ms)
        ids = [row["doc_id"] for row in rows]
        if ids and ids[0] == query["expected_doc_id"]:
            hit_at_1 += 1
        if query["expected_doc_id"] in ids:
            hit_at_5 += 1
        else:
            failures.append(
                {
                    "entity_type": query["entity_type"],
                    "expected_doc_id": query["expected_doc_id"],
                    "query": query["query"][:80],
                    "returned_ids": ids,
                }
            )
    total = len(queries)
    return {
        "query_count": total,
        "top_k": top_k,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "hit_at_1_rate": round(hit_at_1 / total, 4) if total else None,
        "hit_at_5_rate": round(hit_at_5 / total, 4) if total else None,
        "latency_ms": {
            "p50": round(_percentile(latencies_ms, 50) or 0.0, 2),
            "p95": round(_percentile(latencies_ms, 95) or 0.0, 2),
        },
        "failures": failures,
    }


def _gpu_memory() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "device": torch.cuda.get_device_name(0),
            "peak_allocated_mb": round(
                torch.cuda.max_memory_allocated() / 1024 / 1024, 1
            ),
            "peak_reserved_mb": round(
                torch.cuda.max_memory_reserved() / 1024 / 1024, 1
            ),
        }
    except Exception:
        return {}


def run_build_index(
    *,
    manifest_path: Path,
    store_dir: Path,
    embedder: Embedder,
    checkpoint_path: Path | None = None,
    report_path: Path | None = None,
    sample_sizes: dict[str, int] | None = None,
    seed: int = 42,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rebuild: bool = False,
    verify_source: bool = True,
    sanity_queries: int = DEFAULT_SANITY_QUERIES,
    stop_after_batches: int | None = None,
    command: str = "",
    verbose: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    checkpoint_path = checkpoint_path or store_dir / "p4_checkpoint.json"
    report_path = report_path or store_dir / "p4_report.json"
    sample_sizes = dict(sample_sizes or DEFAULT_SAMPLE_SIZES)
    for entity_type in ENTITY_TYPES:
        if int(sample_sizes.get(entity_type, 0)) < 0:
            raise ValueError(f"{entity_type} 抽样数不能为负")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    manifest_path = manifest_path.resolve()
    manifest, sources, manifest_sha256 = _load_source_manifest(
        manifest_path, verify_hashes=verify_source
    )
    samples, actual_sizes = _select_samples(sources, sample_sizes, seed)
    fingerprint = _sample_fingerprint(samples)
    effective_hnsw = dict(HNSW_CONFIGURATION)
    if sum(actual_sizes.values()) <= 100:
        effective_hnsw["sync_threshold"] = 1
    identity = _build_identity(
        embedder=embedder,
        sources=sources,
        manifest_sha256=manifest_sha256,
        requested_sizes=sample_sizes,
        actual_sizes=actual_sizes,
        fingerprint=fingerprint,
        seed=seed,
        batch_size=batch_size,
        hnsw_configuration=effective_hnsw,
    )
    config_id = _config_id(identity)

    old_index: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    resumed = False
    resume_info: dict[str, Any] = {}

    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint_config = checkpoint.get("config_id")
        if checkpoint_config != config_id:
            if not rebuild:
                raise IndexConflict(
                    "已有索引与当前文档/模型/抽样参数不一致；"
                    "请使用 --rebuild 显式重建。"
                )
            old_index = _summarize_old_index(checkpoint)
        else:
            if checkpoint.get("state") == "complete" and not rebuild:
                raise IndexAlreadyComplete(
                    "索引已完整构建；如需覆盖，请使用 --rebuild。"
                )
            if checkpoint.get("state") == "complete" and rebuild:
                old_index = _summarize_old_index(checkpoint)
            if rebuild:
                old_index = old_index or _summarize_old_index(checkpoint)
            else:
                _validate_checkpoint(checkpoint, identity, samples)
                resumed = True

    existing_store = store_dir.exists() and any(
        path.name != "_tmp" for path in store_dir.iterdir()
    )
    if checkpoint is None and existing_store and not rebuild:
        raise IndexConflict(
            "索引目录已有数据但没有 checkpoint；请使用 --rebuild 或更换目录。"
        )
    if checkpoint is None and existing_store and rebuild and old_index is None:
        old_index = {"source": "uncheckpointed", "path": str(store_dir)}

    client = create_client(store_dir)
    if rebuild:
        old_size_before = index_size_bytes(store_dir)
        deleted = delete_collections(client, COLLECTION_NAMES.values())
        clear_client_cache()
        removed_segments = cleanup_orphan_segments(store_dir)
        client = create_client(store_dir)
        if old_index is None:
            old_index = {"source": "uncheckpointed"}
        old_index["size_bytes_before"] = old_size_before
        old_index["deleted_collections"] = deleted
        old_index["removed_orphan_segments"] = removed_segments
        checkpoint = _new_checkpoint(identity, config_id, samples, batch_size)
        _atomic_write_json(checkpoint_path, checkpoint)
    elif checkpoint is None:
        checkpoint = _new_checkpoint(identity, config_id, samples, batch_size)
        _atomic_write_json(checkpoint_path, checkpoint)
    else:
        _verify_resume_ids(client, checkpoint)
        for name, entry in checkpoint["collections"].items():
            resume_info[name] = {
                "processed": len(entry.get("processed_ids") or []),
                "completed_batches": int(entry.get("completed_batches", 0)),
            }
        if verbose:
            processed_total = sum(
                len(entry.get("processed_ids") or [])
                for entry in checkpoint["collections"].values()
            )
            print(f"从 checkpoint 恢复：已完成 {processed_total:,} 条文档", flush=True)

    collection_metadata = {
        "p4_format": FORMAT_VERSION,
        "config_id": config_id,
        "model_id": embedder.model_id,
        "revision": embedder.revision or "",
        "dimension": embedder.dimension,
        "distance": "cosine",
        "ef_search": effective_hnsw["ef_search"],
    }
    collections = {}
    for entity_type in ENTITY_TYPES:
        name = COLLECTION_NAMES[entity_type]
        entry = checkpoint["collections"][name]
        if int(entry.get("total", 0)) == 0:
            continue
        collections[name] = get_or_create_collection(
            client,
            name,
            metadata=collection_metadata,
            configuration={"hnsw": dict(effective_hnsw)},
        )

    embedding_seconds = 0.0
    upsert_seconds = 0.0
    new_batches = 0
    total_documents = sum(len(documents) for documents in samples.values())
    for entity_type in ENTITY_TYPES:
        name = COLLECTION_NAMES[entity_type]
        documents = samples[entity_type]
        entry = checkpoint["collections"][name]
        if not documents:
            continue
        collection = collections[name]
        saved_batch_size = int(entry.get("batch_size", batch_size))
        if saved_batch_size != batch_size:
            batch_size = saved_batch_size
        total_batches = math.ceil(len(documents) / batch_size)
        start_batch = int(entry.get("completed_batches", 0))
        for batch_index in range(start_batch, total_batches):
            batch = documents[
                batch_index * batch_size : (batch_index + 1) * batch_size
            ]
            texts = [document["text"] for document in batch]
            embed_started = time.perf_counter()
            embeddings = embedder.encode_documents(texts)
            embedding_seconds += time.perf_counter() - embed_started
            if embeddings.shape != (len(batch), embedder.dimension):
                raise RuntimeError(
                    f"嵌入维度异常：got={embeddings.shape} "
                    f"expected=({len(batch)}, {embedder.dimension})"
                )

            upsert_started = time.perf_counter()
            upsert_batch(collection, batch, embeddings)
            upsert_seconds += time.perf_counter() - upsert_started

            entry["completed_batches"] = batch_index + 1
            entry["processed_ids"].extend(
                str(document["doc_id"]) for document in batch
            )
            checkpoint["updated_at"] = _now()
            _atomic_write_json(checkpoint_path, checkpoint)
            new_batches += 1
            if verbose and (
                batch_index == total_batches - 1 or (batch_index + 1) % 10 == 0
            ):
                print(
                    f"  {name}: {len(entry['processed_ids']):,}/"
                    f"{len(documents):,} 文档",
                    flush=True,
                )

            if stop_after_batches is not None and new_batches >= stop_after_batches:
                collections.clear()
                try:
                    del collection
                except UnboundLocalError:
                    pass
                del client
                gc.collect()
                clear_client_cache()
                raise IndexInterrupted(
                    f"已在 {new_batches} 个批次后停止；重新运行同一命令可恢复。"
                )

    checkpoint["state"] = "complete"
    checkpoint["completed_at"] = _now()
    checkpoint["updated_at"] = checkpoint["completed_at"]
    _atomic_write_json(checkpoint_path, checkpoint)

    clear_client_cache()
    verify_client = create_client(store_dir)
    reopened_counts: dict[str, int] = {}
    verification_ids: dict[str, dict[str, Any]] = {}
    for entity_type in ENTITY_TYPES:
        name = COLLECTION_NAMES[entity_type]
        expected_ids = [str(document["doc_id"]) for document in samples[entity_type]]
        if not expected_ids:
            reopened_counts[name] = 0
            continue
        collection = verify_client.get_collection(name)
        count = int(collection.count())
        reopened_counts[name] = count
        if count != len(expected_ids):
            raise RuntimeError(
                f"重新打开索引后 {name} 条数不一致："
                f"expected={len(expected_ids)} actual={count}"
            )
        found = existing_ids(collection, expected_ids)
        missing = [doc_id for doc_id in expected_ids if doc_id not in found]
        if missing:
            raise RuntimeError(
                f"重新打开索引后 {name} 缺少 {len(missing)} 条文档"
            )
        verification_ids[name] = {
            "expected": len(expected_ids),
            "verified": len(found),
        }

    sanity = {
        "query_count": 0,
        "top_k": 5,
        "hit_at_1": 0,
        "hit_at_5": 0,
        "hit_at_1_rate": None,
        "hit_at_5_rate": None,
        "latency_ms": {"p50": 0.0, "p95": 0.0},
        "failures": [],
    }
    if sanity_queries > 0:
        queries = _build_sanity_queries(samples, sanity_queries)
        sanity = _run_sanity_checks(verify_client, embedder, queries)

    elapsed = time.perf_counter() - started_at
    documents_report: dict[str, Any] = {}
    for entity_type in ENTITY_TYPES:
        documents = samples[entity_type]
        lengths = [len(document["text"]) for document in documents]
        documents_report[entity_type] = {
            "collection": COLLECTION_NAMES[entity_type],
            "sampled": len(documents),
            "source_lines": sources[entity_type]["expected_lines"],
            "text_length": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
            },
            "reopened_count": reopened_counts[COLLECTION_NAMES[entity_type]],
            "verified_ids": verification_ids.get(COLLECTION_NAMES[entity_type]),
        }

    report = {
        "format": REPORT_VERSION,
        "generated_at": _now(),
        "command": command,
        "mode": "p4_default_10k" if sample_sizes == DEFAULT_SAMPLE_SIZES else "custom",
        "scope": "p4_sample",
        "full_corpus_encoded": False,
        "config_id": config_id,
        "seed": seed,
        "batch_size": batch_size,
        "sample_sizes": sample_sizes,
        "actual_sample_sizes": actual_sizes,
        "documents": documents_report,
        "source": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "files": {
                entity_type: {
                    "path": sources[entity_type]["path"],
                    "sha256": sources[entity_type]["sha256"],
                    "verified_sha256": sources[entity_type]["verified_sha256"],
                    "lines": sources[entity_type]["expected_lines"],
                }
                for entity_type in ENTITY_TYPES
            },
        },
        "embedding": embedder.info(),
        "index": {
            "store_dir": str(store_dir.resolve()),
            "size_bytes": index_size_bytes(store_dir),
            "collections": {
                name: {
                    "count": reopened_counts.get(name, 0),
                    "metadata": collection_metadata,
                }
                for name in COLLECTION_NAMES.values()
                if reopened_counts.get(name, 0) > 0
            },
            "distance": "cosine",
            "hnsw": dict(effective_hnsw),
            "reopened": True,
            "counts_verified": True,
        },
        "performance": {
            "elapsed_sec": round(elapsed, 2),
            "embedding_sec": round(embedding_seconds, 2),
            "upsert_sec": round(upsert_seconds, 2),
            "documents_per_sec_encoding": round(
                total_documents / embedding_seconds, 2
            )
            if embedding_seconds > 0
            else None,
            "documents_per_sec_total": round(total_documents / elapsed, 2)
            if elapsed > 0
            else None,
            "peak_rss_mb": round(peak_rss_mb() or 0.0, 1),
            "gpu": _gpu_memory(),
        },
        "sanity": sanity,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "resumed": resumed,
            "resume_info": resume_info,
            "new_batches_this_run": new_batches,
        },
        "rebuild": {
            "performed": bool(rebuild),
            "old_index": old_index,
        },
    }
    _atomic_write_json(report_path, report)
    report["_report_path"] = str(report_path)
    return report
