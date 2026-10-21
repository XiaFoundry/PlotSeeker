"""Generate a development benchmark from the indexed P4 sample.

These queries are deliberately derived from indexed text to create a
reproducible first baseline.  They are not a substitute for independent human
queries and the generated report must state that limitation.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from config import BENCHMARKS_DIR, DOCUMENTS_DIR, VECTOR_STORE_DIR
from retrieval.keyword_index import load_p4_sample

BENCHMARK_FORMAT = "p6-queries-v1"
DEFAULT_OUTPUT = BENCHMARKS_DIR / "queries.jsonl"
DEFAULT_MANIFEST = BENCHMARKS_DIR / "benchmark_manifest.json"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*")


def _clean_clue(text: str, marker: str, max_chars: int = 180) -> str:
    part = text.split(marker, 1)[1] if marker in text else text
    part = " ".join(part.split())
    sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(part) if item.strip()]
    clue = "".join(sentences[:2]) if sentences else part
    return clue[:max_chars].strip()


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _names(document: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    metadata = document.get("metadata") or {}
    subject_names = []
    if metadata.get("name"):
        subject_names.append(str(metadata["name"]))
    if metadata.get("name_cn"):
        subject_names.append(str(metadata["name_cn"]))
    subject_names.extend(_as_list(metadata.get("aliases")))
    character_names = []
    if document["entity_type"] == "character":
        if metadata.get("name"):
            character_names.append(str(metadata["name"]))
        character_names.extend(_as_list(metadata.get("aliases")))
    episode_names = []
    if document["entity_type"] == "episode":
        for key in ("name", "name_cn"):
            if metadata.get(key):
                episode_names.append(str(metadata[key]))
        subject_names.extend(_as_list(metadata.get("subject_name")))
    if document["entity_type"] == "character":
        subject_names.extend(_as_list(metadata.get("subject_names")))
    return subject_names, character_names, episode_names


def _redact(
    text: str,
    *,
    subject_names: Iterable[str],
    character_names: Iterable[str] = (),
    episode_names: Iterable[str] = (),
) -> str:
    replacements: list[tuple[str, str]] = []
    for value in subject_names:
        replacements.append((str(value), "（某作品）"))
    for value in character_names:
        replacements.append((str(value), "（某角色）"))
    for value in episode_names:
        replacements.append((str(value), "（某章节）"))
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    result = text
    for source, target in replacements:
        if len(source) >= 2:
            result = result.replace(source, target)
    return " ".join(result.split())


def _sentence_fragment(text: str, rng: random.Random, max_chars: int = 120) -> str:
    sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(text) if item.strip()]
    if not sentences:
        return text[:max_chars]
    sentence = rng.choice(sentences)
    return sentence[:max_chars]


def _make_answer(
    query_id: str,
    query: str,
    subject_id: int,
    source_doc_id: str,
    clue_type: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "query": " ".join(query.split()).strip(),
        "subject_id": int(subject_id),
        "acceptable_subject_ids": [int(subject_id)],
        "answerable": True,
        "clue_type": clue_type,
        "source_doc_id": source_doc_id,
        "notes": notes,
    }


def _make_negative(query_id: str, query: str) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "query": query,
        "subject_id": None,
        "acceptable_subject_ids": [],
        "answerable": False,
        "clue_type": "unanswerable",
        "source_doc_id": None,
        "notes": "synthetic negative query",
    }


def _pick_documents(
    documents: list[dict[str, Any]],
    count: int,
    used: set[str],
    predicate: Callable[[dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    if count <= 0:
        return selected
    step = max(1, len(documents) // max(count * 4, 1))
    for index in range(0, len(documents), step):
        document = documents[index]
        doc_id = str(document["doc_id"])
        if doc_id in used or not predicate(document):
            continue
        selected.append(document)
        used.add(doc_id)
        if len(selected) >= count:
            break
    if len(selected) < count:
        for document in documents:
            doc_id = str(document["doc_id"])
            if doc_id in used or not predicate(document):
                continue
            selected.append(document)
            used.add(doc_id)
            if len(selected) >= count:
                break
    return selected


def _convert_traditional(text: str) -> str:
    try:
        import opencc

        return opencc.OpenCC("s2t").convert(text)
    except Exception:
        return text


def build_benchmark(
    *,
    manifest_path: Path = DOCUMENTS_DIR / "manifest.json",
    p4_checkpoint_path: Path = VECTOR_STORE_DIR / "p4_checkpoint.json",
    output_path: Path = DEFAULT_OUTPUT,
    manifest_output: Path = DEFAULT_MANIFEST,
    seed: int = 42,
) -> dict[str, Any]:
    samples, source_info = load_p4_sample(p4_checkpoint_path, manifest_path)
    rng = random.Random(seed)
    for documents in samples.values():
        documents.sort(key=lambda item: str(item["doc_id"]))

    used: set[str] = set()
    queries: list[dict[str, Any]] = []
    seen_queries: set[str] = set()

    def add(entry: dict[str, Any]) -> None:
        query = entry["query"]
        if len(query) < 6 or query in seen_queries:
            return
        seen_queries.add(query)
        entry["query_id"] = f"q{len(queries) + 1:03d}"
        queries.append(entry)

    subject_docs = samples.get("subject", [])
    character_docs = samples.get("character", [])
    episode_docs = samples.get("episode", [])

    def subject_clue(document: dict[str, Any]) -> str:
        subject_names, _, _ = _names(document)
        text = document["text"]
        clue = _clean_clue(text, "简介：")
        redacted = _redact(clue, subject_names=subject_names)
        return redacted if len(redacted) >= 30 else ""

    def character_clue(document: dict[str, Any]) -> str:
        subject_names, character_names, _ = _names(document)
        text = document["text"]
        clue = _clean_clue(text, "角色简介：")
        redacted = _redact(
            clue,
            subject_names=subject_names,
            character_names=character_names,
        )
        return redacted if len(redacted) >= 25 else ""

    def episode_clue(document: dict[str, Any]) -> str:
        subject_names, _, episode_names = _names(document)
        text = document["text"]
        clue = _clean_clue(text, "剧情：")
        redacted = _redact(
            clue,
            subject_names=subject_names,
            episode_names=episode_names,
        )
        return redacted if len(redacted) >= 30 else ""

    for document in _pick_documents(
        subject_docs, 20, used, lambda item: bool(subject_clue(item))
    ):
        clue = subject_clue(document)
        add(
            _make_answer(
                "",
                f"找一部作品，剧情大概是：{clue}",
                document["subject_id"],
                document["doc_id"],
                "subject_summary",
            )
        )

    for document in _pick_documents(
        character_docs, 20, used, lambda item: bool(character_clue(item))
    ):
        clue = character_clue(document)
        add(
            _make_answer(
                "",
                f"我记得有个角色，{clue}。这是哪部作品？",
                document["subject_id"],
                document["doc_id"],
                "character_summary",
            )
        )

    for document in _pick_documents(
        episode_docs, 20, used, lambda item: bool(episode_clue(item))
    ):
        clue = episode_clue(document)
        add(
            _make_answer(
                "",
                f"记得某一话里，{clue}。想找这部作品。",
                document["subject_id"],
                document["doc_id"],
                "episode_plot",
            )
        )

    for document in _pick_documents(
        character_docs, 10, used, lambda item: bool(character_clue(item))
    ):
        clue = character_clue(document)
        add(
            _make_answer(
                "",
                f"找一部作品，里面有个角色：{clue}",
                document["subject_id"],
                document["doc_id"],
                "cross_type_character",
            )
        )

    for document in _pick_documents(
        episode_docs, 10, used, lambda item: bool(episode_clue(item))
    ):
        clue = episode_clue(document)
        add(
            _make_answer(
                "",
                f"想找一部作品，其中有一话：{clue}",
                document["subject_id"],
                document["doc_id"],
                "cross_type_episode",
            )
        )

    name_sources: list[tuple[dict[str, Any], str]] = []
    for document in subject_docs + character_docs:
        metadata = document.get("metadata") or {}
        for key in ("name_cn", "name"):
            value = str(metadata.get(key) or "").strip()
            if len(value) >= 2:
                name_sources.append((document, value))
                break
    for document, name in _pick_name_sources(name_sources, 20, set()):
        add(
            _make_answer(
                "",
                name,
                document["subject_id"],
                document["doc_id"],
                "name",
                notes="exact name query",
            )
        )

    alias_sources: list[tuple[dict[str, Any], str]] = []
    for document in subject_docs + character_docs:
        metadata = document.get("metadata") or {}
        for alias in _as_list(metadata.get("aliases")):
            if len(alias) >= 3:
                alias_sources.append((document, alias))
                break
    for document, alias in _pick_name_sources(alias_sources, 12, set()):
        add(
            _make_answer(
                "",
                alias,
                document["subject_id"],
                document["doc_id"],
                "alias",
                notes="alias query",
            )
        )

    traditional_sources = _pick_documents(
        subject_docs,
        5,
        used,
        lambda item: bool(subject_clue(item)),
    )
    for document in traditional_sources:
        clue = subject_clue(document)
        traditional = _convert_traditional(f"找一部作品，剧情大概是：{clue}")
        add(
            _make_answer(
                "",
                traditional,
                document["subject_id"],
                document["doc_id"],
                "traditional",
                notes="simplified-to-traditional variant",
            )
        )

    fragment_sources = _pick_documents(
        subject_docs + character_docs + episode_docs,
        5,
        used,
        lambda item: len(item.get("text", "")) >= 40,
    )
    for document in fragment_sources:
        if document["entity_type"] == "subject":
            text = _clean_clue(document["text"], "简介：")
            subject_names, character_names, episode_names = _names(document)
        elif document["entity_type"] == "character":
            text = _clean_clue(document["text"], "角色简介：")
            subject_names, character_names, _ = _names(document)
            episode_names = []
        else:
            text = _clean_clue(document["text"], "剧情：")
            subject_names, _, episode_names = _names(document)
            character_names = []
        fragment = _sentence_fragment(
            _redact(
                text,
                subject_names=subject_names,
                character_names=character_names,
                episode_names=episode_names,
            ),
            rng,
        )
        add(
            _make_answer(
                "",
                fragment,
                document["subject_id"],
                document["doc_id"],
                "fragment",
                notes="single-sentence fragment",
            )
        )

    negatives = [
        "找一部主角叫 Zzyzx Qwerty 的动画",
        "有一部作品，剧情是紫色大象在月球开便利店",
        "角色名字是赵不存在，能力是控制时间",
        "想找一部关于会说话的蓝色仙人掌拯救银河系的漫画",
        "有一部作品，主角是一只穿西装的仓鼠侦探",
        "记得一部动画，反派用微波炉统治世界",
        "找一部作品，里面有叫 Qwertyuiop 的魔法学院",
        "有一部关于企鹅参加奥运会的青春故事",
        "想找一部主角能听懂鞋子说话的科幻作品",
        "有一部作品，剧情是月亮被猫偷走了",
    ]
    for query in negatives:
        add(_make_negative("", query))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in queries) + "\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for item in queries:
        counts[item["clue_type"]] = counts.get(item["clue_type"], 0) + 1
    benchmark_manifest = {
        "format": BENCHMARK_FORMAT,
        "generated_at": source_info.get("built_at") or "",
        "seed": seed,
        "query_count": len(queries),
        "answerable_count": sum(1 for item in queries if item["answerable"]),
        "unanswerable_count": sum(1 for item in queries if not item["answerable"]),
        "clue_type_counts": counts,
        "p4_config_id": source_info.get("config_id"),
        "p4_checkpoint": source_info.get("checkpoint_path"),
        "source_manifest": source_info.get("manifest_path"),
        "limitations": [
            "Queries are generated from indexed document text, not independent human queries.",
            "Name and alias queries are intentionally easy and measure exact-match coverage.",
            "Summary/episode/fragment queries may still share wording with indexed text.",
            "Acceptable answers use only the primary subject recorded in the P3 document.",
            "Negative queries are synthetic and may accidentally match real vocabulary.",
        ],
    }
    manifest_output.write_text(
        json.dumps(benchmark_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return benchmark_manifest


def _pick_name_sources(
    sources: list[tuple[dict[str, Any], str]],
    count: int,
    used: set[str],
) -> list[tuple[dict[str, Any], str]]:
    selected: list[tuple[dict[str, Any], str]] = []
    seen_values: set[str] = set()
    for document, value in sources:
        doc_id = str(document["doc_id"])
        if doc_id in used or value in seen_values:
            continue
        selected.append((document, value))
        seen_values.add(value)
        used.add(doc_id)
        if len(selected) >= count:
            break
    return selected
