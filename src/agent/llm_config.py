"""Load the local OpenAI-compatible LLM configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import AGENT_CONFIG_PATH

PROVIDER = "openai-compatible"


class LLMConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_tool_rounds: int
    top_k: int
    provider: str = PROVIDER
    config_path: str = ""

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key.strip())

    def redacted(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "api_key": "***" if self.has_api_key else "",
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_tool_rounds": self.max_tool_rounds,
            "top_k": self.top_k,
            "config_path": self.config_path,
        }


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigError(f"{name} must be an integer") from exc


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigError(f"{name} must be a number") from exc


def load_llm_config(
    path: Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> LLMConfig:
    config_path = Path(path or AGENT_CONFIG_PATH)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"LLM 配置文件不存在：{config_path}。"
            "请复制 config/agent.example.json 为 config/agent.json 并填入 api_key。"
        )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LLMConfigError(f"配置文件 JSON 解析失败：{exc}") from exc
    if not isinstance(payload, dict):
        raise LLMConfigError("配置文件根节点必须是 JSON object")

    api_key = str(payload.get("api_key") or "")
    base_url = str(payload.get("base_url") or "https://api.deepseek.com/v1")
    model = str(payload.get("model") or "deepseek-chat")
    if not api_key:
        api_key = os.environ.get("PLOTSEEKER_LLM_API_KEY", "")
    base_url = os.environ.get("PLOTSEEKER_LLM_BASE_URL", base_url)
    model = os.environ.get("PLOTSEEKER_LLM_MODEL", model)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key == "api_key":
            api_key = str(value)
        elif key == "base_url":
            base_url = str(value)
        elif key == "model":
            model = str(value)

    if not base_url.startswith(("http://", "https://")):
        raise LLMConfigError("base_url must start with http:// or https://")
    if not model.strip():
        raise LLMConfigError("model must not be empty")

    config = LLMConfig(
        api_key=api_key.strip(),
        base_url=base_url.strip(),
        model=model.strip(),
        temperature=_as_float(payload.get("temperature", 0.2), "temperature"),
        max_tokens=_as_int(payload.get("max_tokens", 1200), "max_tokens"),
        timeout_seconds=_as_float(
            payload.get("timeout_seconds", 90), "timeout_seconds"
        ),
        max_tool_rounds=_as_int(
            payload.get("max_tool_rounds", 4), "max_tool_rounds"
        ),
        top_k=_as_int(payload.get("top_k", 5), "top_k"),
        provider=str(payload.get("provider") or PROVIDER),
        config_path=str(config_path.resolve()),
    )
    if not 0.0 <= config.temperature <= 2.0:
        raise LLMConfigError("temperature must be between 0 and 2")
    if config.max_tokens <= 0:
        raise LLMConfigError("max_tokens must be positive")
    if config.timeout_seconds <= 0:
        raise LLMConfigError("timeout_seconds must be positive")
    if not 1 <= config.max_tool_rounds <= 10:
        raise LLMConfigError("max_tool_rounds must be between 1 and 10")
    if not 1 <= config.top_k <= 20:
        raise LLMConfigError("top_k must be between 1 and 20")
    return config
