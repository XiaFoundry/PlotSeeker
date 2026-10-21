"""Build and query the P5 SQLite FTS5 keyword index.

The index is built from the exact document IDs recorded in the P4 checkpoint,
so keyword and vector retrieval cover the same 10,000-document sample.  Two
tokenizers are indexed in parallel for a small comparison; the winner is stored
as the default in the index metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from config import STORAGE_DIR
from retrieval.models import DocumentHit
from retrieval.preprocess import (
    DEFAULT_TOKENIZER,
    TOKENIZERS,
    fts_query,
    make_snippet,
    token_text,
    tokenize_query,
)

KEYWORD_DB_PATH = STORAGE_DIR / "fts" / "p5_keyword_10k.db"
KEYWORD_REPORT_PATH = STORAGE_DIR / "fts" / "p5_keyword_report.json"
FTS_TABLES = {"jieba": "fts_jieba", "bigram": "fts_bigram"}
BUILD_FORMAT = "p5-keyword-v1"
BUILD_BATCH_SIZE = 500


class KeywordIndexError(RuntimeError):
    pass


class KeywordIndexExists(FileExistsError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"P4 checkpoint 不存在：{path}")
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("state") != "complete":
        raise KeywordIndexError("P4 索引尚未完成，不能构建关键词索引")
    if not checkpoint.get("identity") or not checkpoint.get("collections"):
        raise KeywordIndexError("P4 checkpoint 缺少 identity/collections")
    return checkpoint


def _verify_p4_sources(
    checkpoint: dict[str, Any], manifest_path: Path, *, verify_hashes: bool
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"P3 manifest 不存在：{manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    identity = checkpoint["identity"]
    expected_manifest = identity.get("manifest_sha256")
    if expected_manifest and manifest_sha != expected_manifest:
        raise KeywordIndexError(
            f"P3 manifest 与 P4 checkpoint 不一致：{manifest_sha} != {expected_manifest}"
        )
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    files = manifest.get("files") or {}
    sources: dict[str, Any] = {}
    for entity_type, expected in (identity.get("sources") or {}).items():
        entry = files.get(entity_type)
        if not isinstance(entry, dict):
            raise KeywordIndexError(f"manifest 缺少 {entity_type} 文件信息")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if verify_hashes:
            actual = _sha256_file(path)
            if expected.get("sha256") and actual != expected["sha256"]:
                raise KeywordIndexError(
                    f"{entity_type} 文档哈希与 P4 checkpoint 不一致"
                )
        sources[entity_type] = {
            "path": str(path.resolve()),
            "sha256": expected.get("sha256"),
            "lines": expected.get("lines"),
        }
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "sources": sources,
    }


def load_p4_sample(
    checkpoint_path: Path,
    manifest_path: Path,
    *,
    verify_source: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    checkpoint = _load_checkpoint(checkpoint_path)
    source_info = _verify_p4_sources(
        checkpoint, manifest_path, verify_hashes=verify_source
    )
    identity = checkpoint["identity"]
    collection_to_type = {
        name: entity_type for entity_type, name in identity["collections"].items()
    }
    wanted: dict[str, str] = {}
    expected_counts: dict[str, int] = {}
    for collection_name, entry in checkpoint["collections"].items():
        entity_type = collection_to_type.get(collection_name)
        if entity_type is None:
            raise KeywordIndexError(f"未知 collection：{collection_name}")
        ids = [str(value) for value in entry.get("processed_ids") or []]
        expected_counts[entity_type] = len(ids)
        for doc_id in ids:
            wanted[doc_id] = entity_type
    if not wanted:
        raise KeywordIndexError("P4 checkpoint 中没有可用的文档 ID")

    samples: dict[str, list[dict[str, Any]]] = {
        entity_type: [] for entity_type in expected_counts
    }
    found: set[str] = set()
    files = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    for entity_type, entry in files.items():
        path = Path(entry["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                document = json.loads(line)
                doc_id = str(document.get("doc_id", ""))
                if doc_id in wanted:
                    samples[wanted[doc_id]].append(document)
                    found.add(doc_id)
    missing = sorted(set(wanted) - found)
    if missing:
        raise KeywordIndexError(
            f"{len(missing)} 个 P4 文档 ID 在 P3 文件中缺失，例如 {missing[:3]}"
        )
    for entity_type, documents in samples.items():
        documents.sort(key=lambda item: str(item["doc_id"]))
        if len(documents) != expected_counts[entity_type]:
            raise KeywordIndexError(
                f"{entity_type} 文档数不一致：{len(documents)} != "
                f"{expected_counts[entity_type]}"
            )
    info = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "config_id": checkpoint.get("config_id"),
        "p4_identity": identity,
        **source_info,
    }
    return samples, info


def _create_schema(connection: sqlite3.Connection, tokenizers: Sequence[str]) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = FILE;
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            entity_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            tokens_jieba TEXT NOT NULL DEFAULT '',
            tokens_bigram TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX idx_documents_entity_type ON documents(entity_type);
        CREATE INDEX idx_documents_subject_id ON documents(subject_id);
        CREATE TABLE index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    for tokenizer in tokenizers:
        table = FTS_TABLES[tokenizer]
        connection.execute(
            f"CREATE VIRTUAL TABLE {table} USING fts5("
            "doc_id UNINDEXED, entity_type UNINDEXED, tokens, "
            "tokenize='unicode61')"
        )


def _insert_documents(
    connection: sqlite3.Connection,
    documents: Sequence[dict[str, Any]],
    tokenizers: Sequence[str],
) -> dict[str, dict[str, int]]:
    stats = {
        tokenizer: {"tokens": 0, "min": 0, "max": 0, "documents": 0}
        for tokenizer in tokenizers
    }
    for start in range(0, len(documents), BUILD_BATCH_SIZE):
        batch = documents[start : start + BUILD_BATCH_SIZE]
        connection.execute("BEGIN")
        for document in batch:
            token_map = {
                tokenizer: token_text(str(document["text"]), tokenizer).split()
                for tokenizer in tokenizers
            }
            connection.execute(
                "INSERT INTO documents("
                "doc_id, entity_type, entity_id, subject_id, text, metadata_json, "
                "tokens_jieba, tokens_bigram"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(document["doc_id"]),
                    str(document["entity_type"]),
                    int(document["entity_id"]),
                    int(document.get("subject_id") or -1),
                    str(document["text"]),
                    json.dumps(
                        document.get("metadata") or {}, ensure_ascii=False
                    ),
                    " ".join(token_map.get("jieba", [])),
                    " ".join(token_map.get("bigram", [])),
                ),
            )
            for tokenizer in tokenizers:
                tokens = token_map[tokenizer]
                table = FTS_TABLES[tokenizer]
                connection.execute(
                    f"INSERT INTO {table}(doc_id, entity_type, tokens) VALUES (?, ?, ?)",
                    (
                        str(document["doc_id"]),
                        str(document["entity_type"]),
                        " ".join(tokens),
                    ),
                )
                current = stats[tokenizer]
                count = len(tokens)
                current["documents"] += 1
                current["tokens"] += count
                if count:
                    if current["min"] == 0 or count < current["min"]:
                        current["min"] = count
                    if count > current["max"]:
                        current["max"] = count
        connection.commit()
    return stats


def _search_fts(
    connection: sqlite3.Connection,
    tokenizer: str,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    tokens = tokenize_query(query, tokenizer)
    query_text = fts_query(tokens)
    if not query_text:
        return []
    table = FTS_TABLES[tokenizer]
    rows = connection.execute(
        f"""
        SELECT f.doc_id, f.entity_type, bm25({table}) AS score
        FROM {table} AS f
        WHERE {table} MATCH ?
        ORDER BY score ASC
        LIMIT ?
        """,
        (query_text, max(1, int(top_k))),
    ).fetchall()
    return [
        {"doc_id": row[0], "entity_type": row[1], "score": float(row[2])}
        for row in rows
    ]


def _comparison_queries(
    samples: dict[str, list[dict[str, Any]]], limit_per_type: int = 80
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for entity_type, documents in samples.items():
        selected = documents
        if len(selected) > limit_per_type:
            step = len(selected) / limit_per_type
            selected = [selected[int(index * step)] for index in range(limit_per_type)]
        for document in selected:
            metadata = document.get("metadata") or {}
            name = str(metadata.get("name_cn") or metadata.get("name") or "").strip()
            if name:
                queries.append(
                    {
                        "query": name,
                        "expected_doc_id": document["doc_id"],
                        "entity_type": entity_type,
                        "kind": "name",
                    }
                )
            snippet = str(document.get("text") or "")[:80].strip()
            if snippet:
                queries.append(
                    {
                        "query": snippet,
                        "expected_doc_id": document["doc_id"],
                        "entity_type": entity_type,
                        "kind": "snippet",
                    }
                )
    return queries


def _compare_tokenizers(
    connection: sqlite3.Connection,
    samples: dict[str, list[dict[str, Any]]],
    tokenizers: Sequence[str],
) -> dict[str, Any]:
    queries = _comparison_queries(samples)
    report: dict[str, Any] = {}
    for tokenizer in tokenizers:
        hits5 = 0
        reciprocal_rank = 0.0
        evaluated = 0
        for query in queries:
            rows = _search_fts(connection, tokenizer, query["query"], 10)
            if not rows:
                evaluated += 1
                continue
            evaluated += 1
            ids = [row["doc_id"] for row in rows]
            if query["expected_doc_id"] in ids[:5]:
                hits5 += 1
            if query["expected_doc_id"] in ids:
                reciprocal_rank += 1.0 / (ids.index(query["expected_doc_id"]) + 1)
        report[tokenizer] = {
            "queries": evaluated,
            "recall_at_5": round(hits5 / evaluated, 4) if evaluated else None,
            "mrr_at_10": round(reciprocal_rank / evaluated, 4) if evaluated else None,
        }
    return report


def _choose_default(comparison: dict[str, Any], tokenizers: Sequence[str]) -> str:
    def key(name: str) -> tuple[float, float, int]:
        item = comparison.get(name) or {}
        recall = item.get("recall_at_5") or 0.0
        mrr = item.get("mrr_at_10") or 0.0
        preference = 1 if name == DEFAULT_TOKENIZER else 0
        return (recall, mrr, preference)

    return max(tokenizers, key=key)


def _write_metadata(
    connection: sqlite3.Connection, values: dict[str, Any]
) -> None:
    for key, value in values.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        connection.execute(
            "INSERT INTO index_metadata(key, value) VALUES (?, ?)", (key, value)
        )


def build_keyword_index(
    *,
    manifest_path: Path,
    p4_checkpoint_path: Path,
    output_path: Path = KEYWORD_DB_PATH,
    report_path: Path = KEYWORD_REPORT_PATH,
    tokenizers: Sequence[str] = TOKENIZERS,
    rebuild: bool = False,
    verify_source: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    tokenizers = tuple(tokenizers)
    for tokenizer in tokenizers:
        if tokenizer not in FTS_TABLES:
            raise ValueError(f"unknown tokenizer: {tokenizer}")
    if not tokenizers:
        raise ValueError("at least one tokenizer is required")
    if output_path.exists() and not rebuild:
        raise KeywordIndexExists(
            f"关键词索引已存在：{output_path}；使用 --rebuild 显式重建"
        )

    started = time.perf_counter()
    samples, source_info = load_p4_sample(
        p4_checkpoint_path, manifest_path, verify_source=verify_source
    )
    documents = [document for items in samples.values() for document in items]
    documents.sort(key=lambda item: str(item["doc_id"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".building")
    if temp_path.exists():
        temp_path.unlink()

    connection = sqlite3.connect(temp_path)
    try:
        _create_schema(connection, tokenizers)
        token_stats = _insert_documents(connection, documents, tokenizers)
        comparison = _compare_tokenizers(connection, samples, tokenizers)
        default_tokenizer = _choose_default(comparison, tokenizers)
        _write_metadata(
            connection,
            {
                "format": BUILD_FORMAT,
                "built_at": _now(),
                "p4_checkpoint": source_info["checkpoint_path"],
                "p4_config_id": source_info.get("config_id"),
                "manifest_sha256": source_info["manifest_sha256"],
                "tokenizers": list(tokenizers),
                "default_tokenizer": default_tokenizer,
                "counts": {key: len(value) for key, value in samples.items()},
                "token_stats": token_stats,
                "tokenizer_comparison": comparison,
            },
        )
        connection.commit()
    except Exception:
        connection.close()
        temp_path.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    size_bytes = temp_path.stat().st_size
    os.replace(temp_path, output_path)

    verify_connection = sqlite3.connect(
        f"file:{output_path}?mode=ro", uri=True
    )
    try:
        doc_count = verify_connection.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
        fts_counts = {
            tokenizer: verify_connection.execute(
                f"SELECT COUNT(*) FROM {FTS_TABLES[tokenizer]}"
            ).fetchone()[0]
            for tokenizer in tokenizers
        }
    finally:
        verify_connection.close()
    if doc_count != len(documents):
        raise KeywordIndexError(
            f"关键词索引条数校验失败：{doc_count} != {len(documents)}"
        )

    elapsed = time.perf_counter() - started
    report = {
        "format": BUILD_FORMAT,
        "generated_at": _now(),
        "output_path": str(output_path.resolve()),
        "size_bytes": size_bytes,
        "elapsed_sec": round(elapsed, 2),
        "document_count": doc_count,
        "counts": {key: len(value) for key, value in samples.items()},
        "tokenizers": list(tokenizers),
        "default_tokenizer": default_tokenizer,
        "token_stats": token_stats,
        "tokenizer_comparison": comparison,
        "verified": {
            "documents": doc_count,
            "fts_counts": fts_counts,
        },
        "source": source_info,
    }
    _atomic_write_json(report_path, report)
    if verbose:
        print(
            f"关键词索引：{output_path}  {len(documents):,} 文档  "
            f"{size_bytes / 1024 / 1024:.1f} MiB  "
            f"默认分词={default_tokenizer}  耗时 {elapsed:.1f}s"
        )
    return report


class KeywordSearcher:
    def __init__(self, db_path: Path = KEYWORD_DB_PATH, tokenizer: str | None = None):
        self.db_path = Path(db_path)
        if not self.db_path.is_file():
            raise FileNotFoundError(f"关键词索引不存在：{self.db_path}")
        self._connection = sqlite3.connect(
            f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False
        )
        metadata = self.metadata()
        self.default_tokenizer = metadata.get("default_tokenizer", DEFAULT_TOKENIZER)
        self.tokenizer = tokenizer or self.default_tokenizer
        if self.tokenizer not in FTS_TABLES:
            raise ValueError(f"unknown tokenizer: {self.tokenizer}")
        table = FTS_TABLES[self.tokenizer]
        exists = self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            raise KeywordIndexError(f"关键词索引缺少表：{table}")

    def metadata(self) -> dict[str, Any]:
        rows = self._connection.execute(
            "SELECT key, value FROM index_metadata"
        ).fetchall()
        result: dict[str, Any] = {}
        for key, value in rows:
            try:
                result[key] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                result[key] = value
        return result

    def document(self, doc_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT doc_id, entity_type, entity_id, subject_id, text, metadata_json "
            "FROM documents WHERE doc_id=?",
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "doc_id": row[0],
            "entity_type": row[1],
            "entity_id": row[2],
            "subject_id": row[3],
            "text": row[4],
            "metadata": json.loads(row[5]),
        }

    def search(
        self,
        query: str,
        top_k: int = 50,
        entity_types: Iterable[str] | None = None,
    ) -> list[DocumentHit]:
        tokens = tokenize_query(query, self.tokenizer)
        query_text = fts_query(tokens)
        if not query_text:
            return []
        table = FTS_TABLES[self.tokenizer]
        sql = (
            f"SELECT f.doc_id, bm25({table}) AS score, d.entity_type, "
            "d.entity_id, d.subject_id, d.text, d.metadata_json, "
            f"d.tokens_{self.tokenizer} "
            f"FROM {table} AS f JOIN documents AS d ON d.doc_id=f.doc_id "
            f"WHERE {table} MATCH ?"
        )
        params: list[Any] = [query_text]
        types = list(entity_types or [])
        if types:
            placeholders = ",".join("?" for _ in types)
            sql += f" AND d.entity_type IN ({placeholders})"
            params.extend(types)
        sql += " ORDER BY score ASC LIMIT ?"
        params.append(max(1, int(top_k)))
        rows = self._connection.execute(sql, params).fetchall()
        hits: list[DocumentHit] = []
        query_token_set = set(tokens)
        for rank, row in enumerate(rows, 1):
            metadata = json.loads(row[6])
            doc_tokens = set(str(row[7] or "").split())
            coverage = (
                len(query_token_set & doc_tokens) / len(query_token_set)
                if query_token_set
                else 0.0
            )
            hits.append(
                DocumentHit(
                    doc_id=row[0],
                    entity_type=row[2],
                    entity_id=int(row[3]),
                    subject_id=int(row[4]),
                    text=row[5],
                    metadata=metadata,
                    score=float(row[1]),
                    rank=rank,
                    source="keyword",
                    snippet=make_snippet(row[5], tokens),
                    coverage=coverage,
                )
            )
        return hits

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "KeywordSearcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
