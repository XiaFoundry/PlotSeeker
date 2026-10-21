"""Thin Chroma persistence helpers used by P4 and later retrieval stages."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


class IndexConflict(RuntimeError):
    """Raised when an existing index cannot be safely reused or overwritten."""


def _prepare_temp_dir(store_dir: Path) -> Path:
    """Point SQLite/Chroma temporary files at a writable project directory.

    Chroma 1.5.x's Rust bindings create temporary files during migrations.
    The default Windows TEMP directory can be ACL-restricted in sandboxed
    sessions, which surfaces as SQLite error 14 (unable to open database file).
    """

    temp_dir = store_dir / "_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for name in ("TEMP", "TMP", "SQLITE_TMPDIR"):
        os.environ[name] = str(temp_dir)
    return temp_dir


def create_client(store_dir: Path, *, allow_reset: bool = False):
    from chromadb import PersistentClient
    from chromadb.config import Settings

    store_dir.mkdir(parents=True, exist_ok=True)
    _prepare_temp_dir(store_dir)
    settings = Settings(
        anonymized_telemetry=False,
        allow_reset=allow_reset,
        is_persistent=True,
        persist_directory=str(store_dir),
    )
    return PersistentClient(path=str(store_dir), settings=settings)


def clear_client_cache() -> None:
    from chromadb.api.shared_system_client import SharedSystemClient

    SharedSystemClient.clear_system_cache()


def collection_names(client) -> set[str]:
    return {collection.name for collection in client.list_collections()}


def collection_exists(client, name: str) -> bool:
    return name in collection_names(client)


def get_or_create_collection(
    client,
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
    configuration: dict[str, Any] | None = None,
):
    existing = {collection.name for collection in client.list_collections()}
    if name in existing:
        return client.get_collection(name)
    return client.create_collection(
        name=name,
        configuration=configuration or {"hnsw": {"space": "cosine"}},
        metadata=metadata,
        embedding_function=None,
    )


def delete_collections(client, names: Iterable[str]) -> list[str]:
    existing = collection_names(client)
    deleted: list[str] = []
    for name in names:
        if name in existing:
            client.delete_collection(name)
            deleted.append(name)
    return deleted


_SEGMENT_DIR_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def cleanup_orphan_segments(store_dir: Path) -> list[str]:
    """Remove HNSW directories no longer referenced by the Chroma database.

    Deleting a collection removes its rows from ``segments`` but Chroma 1.5.x
    leaves the on-disk HNSW directory behind.  Only UUID-named directories that
    are absent from the live segment table are removed.
    """

    database_path = store_dir / "chroma.sqlite3"
    if not database_path.is_file():
        return []
    try:
        connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        try:
            active = {
                str(row[0])
                for row in connection.execute("SELECT id FROM segments")
            }
        finally:
            connection.close()
    except sqlite3.Error:
        return []

    removed: list[str] = []
    for path in store_dir.iterdir():
        if not path.is_dir() or path.name == "_tmp":
            continue
        if not _SEGMENT_DIR_RE.match(path.name):
            continue
        if path.name in active:
            continue
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            removed.append(path.name)
    return removed


def _scalar_metadata(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def document_metadata(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata") or {}
    result: dict[str, Any] = {
        "doc_id": document["doc_id"],
        "entity_type": document["entity_type"],
        "entity_id": int(document["entity_id"]),
        "subject_id": int(document["subject_id"])
        if document.get("subject_id") is not None
        else -1,
        "source": document.get("source", ""),
    }
    for key, value in metadata.items():
        cleaned = _scalar_metadata(value)
        if cleaned is not None:
            result[key] = cleaned
    return result


def upsert_batch(
    collection,
    documents: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
) -> None:
    if len(documents) != len(embeddings):
        raise ValueError("documents and embeddings must have the same length")
    if not documents:
        return
    collection.upsert(
        ids=[document["doc_id"] for document in documents],
        embeddings=embeddings.astype(np.float32).tolist(),
        documents=[document["text"] for document in documents],
        metadatas=[document_metadata(document) for document in documents],
    )


def existing_ids(collection, ids: Sequence[str], chunk_size: int = 1000) -> set[str]:
    found: set[str] = set()
    for start in range(0, len(ids), chunk_size):
        chunk = list(ids[start : start + chunk_size])
        result = collection.get(ids=chunk, include=[])
        found.update(result.get("ids") or [])
    return found


def query_collection(
    collection,
    embedding: np.ndarray,
    top_k: int,
    *,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    result = collection.query(
        query_embeddings=embedding.astype(np.float32).reshape(1, -1).tolist(),
        n_results=max(1, int(top_k)),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    rows: list[dict[str, Any]] = []
    for index, doc_id in enumerate(ids):
        rows.append(
            {
                "doc_id": doc_id,
                "document": documents[index] if index < len(documents) else None,
                "metadata": metadatas[index] if index < len(metadatas) else None,
                "distance": distances[index] if index < len(distances) else None,
            }
        )
    return rows


def index_size_bytes(store_dir: Path) -> int:
    if not store_dir.exists():
        return 0
    total = 0
    for root, dirs, files in os.walk(store_dir):
        root_path = Path(root)
        if "_tmp" in root_path.parts:
            continue
        for filename in files:
            try:
                total += (root_path / filename).stat().st_size
            except OSError:
                continue
    return total
