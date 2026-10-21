"""Shared Chinese normalization and tokenization for indexing and queries."""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache

from config import STORAGE_DIR

DEFAULT_TOKENIZER = "jieba"
TOKENIZERS = ("jieba", "bigram")
MAX_QUERY_TOKENS = 32
STOPWORDS = frozenset(
    {
        "的",
        "了",
        "在",
        "是",
        "和",
        "与",
        "及",
        "或",
        "也",
        "都",
        "就",
        "而",
        "被",
        "把",
        "让",
        "给",
        "对",
        "从",
        "到",
        "为",
        "中",
        "上",
        "下",
        "吗",
        "呢",
        "啊",
        "呀",
        "吧",
        "我",
        "你",
        "他",
        "她",
        "它",
        "这",
        "那",
        "有",
        "一个",
        "一种",
        "一些",
        "以及",
        "因为",
        "所以",
        "但是",
        "如果",
        "然后",
        "可以",
        "没有",
        "不是",
        "就是",
        "还是",
        "这个",
        "那个",
        "什么",
        "怎么",
        "如何",
        "自己",
        "の",
        "と",
        "が",
        "を",
        "に",
        "は",
        "で",
        "も",
        "や",
        "から",
        "まで",
        "より",
        "へ",
        "找",
        "作品",
        "剧情",
        "大概",
        "这是",
        "哪部",
        "想找",
        "记得",
        "里面",
        "有个",
        "角色",
        "章节",
        "一话",
        "某一",
        "一部",
        "their",
        "there",
        "this",
        "that",
        "these",
        "those",
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "about",
        "after",
        "before",
        "was",
        "were",
        "his",
        "her",
        "its",
        "are",
        "is",
        "it",
        "as",
        "at",
        "by",
        "on",
        "or",
        "of",
        "to",
        "in",
    }
)

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_LATIN_RE = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=1)
def _opencc_converter():
    try:
        import opencc

        return opencc.OpenCC("t2s")
    except Exception:
        return None


def normalize_text(text: str, *, to_simplified: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text).lower()
    if to_simplified:
        converter = _opencc_converter()
        if converter is not None:
            try:
                normalized = converter.convert(normalized)
            except Exception:
                pass
    chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith(("P", "Z", "S")) or char in {"\u200b", "\ufeff"}:
            chars.append(" ")
        else:
            chars.append(char)
    return " ".join("".join(chars).split())


@lru_cache(maxsize=1)
def _jieba():
    import jieba

    cache_dir = STORAGE_DIR / "cache" / "jieba"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jieba.dt.tmp_dir = str(cache_dir)
    jieba.setLogLevel(logging.ERROR)
    return jieba


def tokenize_jieba(text: str, *, to_simplified: bool = True) -> list[str]:
    normalized = normalize_text(text, to_simplified=to_simplified)
    if not normalized:
        return []
    jieba = _jieba()
    tokens = [
        token.strip()
        for token in jieba.lcut_for_search(normalized)
        if token.strip() and not token.isspace()
    ]
    return tokens


def _tokenize_runs(normalized: str) -> list[str]:
    tokens: list[str] = []
    for run in normalized.split():
        if not run:
            continue
        latin_matches = list(_LATIN_RE.finditer(run))
        if latin_matches and len(latin_matches) == 1 and latin_matches[0].span() == (
            0,
            len(run),
        ):
            tokens.append(run)
            continue
        for match in _CJK_RE.finditer(run):
            cjk = match.group(0)
            if len(cjk) == 1:
                tokens.append(cjk)
            else:
                tokens.extend(cjk[index : index + 2] for index in range(len(cjk) - 1))
        for match in _LATIN_RE.finditer(run):
            tokens.append(match.group(0))
    return tokens


def tokenize_bigram(text: str, *, to_simplified: bool = True) -> list[str]:
    return _tokenize_runs(normalize_text(text, to_simplified=to_simplified))


def tokenize(
    text: str, tokenizer: str = DEFAULT_TOKENIZER, *, to_simplified: bool = True
) -> list[str]:
    if tokenizer == "jieba":
        return tokenize_jieba(text, to_simplified=to_simplified)
    if tokenizer == "bigram":
        return tokenize_bigram(text, to_simplified=to_simplified)
    raise ValueError(f"unknown tokenizer: {tokenizer}")


def _prepare_query_tokens(tokens: list[str]) -> list[str]:
    prepared: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token or token in STOPWORDS:
            continue
        if token in seen:
            continue
        prepared.append(token)
        seen.add(token)
        if len(prepared) >= MAX_QUERY_TOKENS:
            break
    return prepared


def tokenize_query(
    text: str, tokenizer: str = DEFAULT_TOKENIZER, *, to_simplified: bool = True
) -> list[str]:
    return _prepare_query_tokens(
        tokenize(text, tokenizer, to_simplified=to_simplified)
    )


def token_text(text: str, tokenizer: str) -> str:
    return " ".join(tokenize(text, tokenizer))


def fts_query(tokens: list[str]) -> str:
    cleaned = []
    for token in tokens:
        value = token.replace('"', '""').strip()
        if value:
            cleaned.append(f'"{value}"')
    return " OR ".join(cleaned)


def make_snippet(text: str, tokens: list[str], width: int = 160) -> str:
    if not text:
        return ""
    if width <= 0:
        return ""
    lowered = text.lower()
    for token in tokens:
        needle = token.lower().strip()
        if not needle:
            continue
        position = lowered.find(needle)
        if position >= 0:
            start = max(0, position - width // 3)
            end = min(len(text), start + width)
            snippet = text[start:end].strip()
            if start > 0:
                snippet = "…" + snippet
            if end < len(text):
                snippet += "…"
            return snippet
    snippet = text[:width].strip()
    return snippet + ("…" if len(text) > width else "")
