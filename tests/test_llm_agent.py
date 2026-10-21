from __future__ import annotations

import json

import pytest

from agent.llm_agent import LLMAgent, LLMAPIError, OpenAICompatibleClient, openai_tool_definitions
from agent.llm_config import LLMConfig, LLMConfigError, load_llm_config
from agent.tools import AgentTools
from embeddings.fake import DeterministicFakeEmbedder


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        if not self.responses:
            raise LLMAPIError("no fake response left")
        return self.responses.pop(0)

    def close(self):
        pass


def _config(**overrides):
    values = {
        "api_key": "test-key",
        "base_url": "http://localhost:9999/v1",
        "model": "fake-model",
        "temperature": 0.0,
        "max_tokens": 256,
        "timeout_seconds": 10.0,
        "max_tool_rounds": 3,
        "top_k": 3,
    }
    values.update(overrides)
    return LLMConfig(**values)


def _tool_agent(retrieval_env, responses, config=None):
    tools = AgentTools(
        keyword_db=retrieval_env["keyword_db"],
        store_dir=retrieval_env["store_dir"],
        checkpoint_path=retrieval_env["checkpoint"],
        sqlite_path=retrieval_env["sqlite"],
        embedder=DeterministicFakeEmbedder(retrieval_env["fake_dimension"]),
    )
    client = FakeClient(responses)
    agent = LLMAgent(tools, config or _config(), client=client)
    return tools, agent, client


def test_config_load_and_env_override(tmp_path, monkeypatch):
    path = tmp_path / "agent.json"
    path.write_text(
        json.dumps(
            {
                "api_key": "",
                "base_url": "https://example.com/v1",
                "model": "file-model",
                "temperature": 0.2,
                "max_tokens": 100,
                "timeout_seconds": 20,
                "max_tool_rounds": 3,
                "top_k": 5,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PLOTSEEKER_LLM_API_KEY", "env-key")
    monkeypatch.setenv("PLOTSEEKER_LLM_MODEL", "env-model")
    config = load_llm_config(path)
    assert config.api_key == "env-key"
    assert config.model == "env-model"
    assert config.endpoint == "https://example.com/v1/chat/completions"
    assert config.redacted()["api_key"] == "***"
    overridden = load_llm_config(
        path, overrides={"base_url": "http://127.0.0.1:11434/v1"}
    )
    assert overridden.base_url == "http://127.0.0.1:11434/v1"


def test_config_missing_file_and_invalid_values(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_llm_config(tmp_path / "missing.json")
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"base_url": "not-a-url", "model": "x"}), encoding="utf-8"
    )
    with pytest.raises(LLMConfigError):
        load_llm_config(path)


def test_openai_tool_definitions():
    definitions = openai_tool_definitions()
    names = {item["function"]["name"] for item in definitions}
    assert names == {
        "search_subjects",
        "search_characters",
        "search_episodes",
        "get_related_subjects",
        "ask_user_for_clarification",
    }
    assert all(item["type"] == "function" for item in definitions)


def test_client_requires_api_key():
    client = OpenAICompatibleClient(_config(api_key=""))
    try:
        with pytest.raises(LLMAPIError):
            client.chat([{"role": "user", "content": "x"}])
    finally:
        client.close()


class FakeHTTPResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeHTTPClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.response

    def close(self):
        pass


def test_openai_client_payload_and_headers():
    response = FakeHTTPResponse(
        payload={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"total_tokens": 3},
        }
    )
    client = OpenAICompatibleClient(_config())
    client._client = FakeHTTPClient(response)
    try:
        data = client.chat(
            [{"role": "user", "content": "hello"}],
            tools=openai_tool_definitions(),
        )
    finally:
        client.close()
    assert data["choices"][0]["message"]["content"] == "ok"
    call = client._client.calls[0]
    assert call["url"].endswith("/chat/completions")
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["json"]["model"] == "fake-model"
    assert call["json"]["tool_choice"] == "auto"
    assert call["json"]["tools"][0]["type"] == "function"


def test_openai_client_http_error_redacts_key():
    response = FakeHTTPResponse(
        status_code=401, text="bad key test-key leaked"
    )
    client = OpenAICompatibleClient(_config())
    client._client = FakeHTTPClient(response)
    try:
        with pytest.raises(LLMAPIError) as exc_info:
            client.chat([{"role": "user", "content": "hello"}])
    finally:
        client.close()
    assert "401" in str(exc_info.value)
    assert "test-key" not in str(exc_info.value)
    assert "***" in str(exc_info.value)


def test_llm_agent_tool_call_flow(retrieval_env):
    first = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_characters",
                                "arguments": json.dumps(
                                    {"query": "感知时间循环", "top_k": 3}
                                ),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    second = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "最可能是《时间循环学园》（subject:1）。",
                }
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 8,
            "total_tokens": 28,
        },
    }
    tools, agent, client = _tool_agent(retrieval_env, [first, second])
    try:
        result = agent.run("有个角色能感知时间循环")
    finally:
        agent.close()
        tools.close()
    assert result["used_llm"] is True
    assert result["answer"].startswith("最可能是")
    assert result["rounds"] == 1
    assert result["tool_calls"][0]["tool"] == "search_characters"
    assert result["tool_calls"][0]["count"] > 0
    assert result["usage"]["total_tokens"] == 43
    assert len(client.calls) == 2
    tool_message = client.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "character:1" in tool_message["content"]


def test_llm_agent_retries_if_no_tool_called(retrieval_env):
    first = {
        "choices": [{"message": {"role": "assistant", "content": "unverified"}}]
    }
    second = {
        "choices": [{"message": {"role": "assistant", "content": "still unverified"}}]
    }
    tools, agent, client = _tool_agent(retrieval_env, [first, second])
    try:
        result = agent.run("随便问问")
    finally:
        agent.close()
        tools.close()
    assert result["tool_calls"] == []
    assert "llm_did_not_call_tools" in result["warnings"]
    assert len(client.calls) == 2


def test_llm_agent_clarification_tool(retrieval_env):
    first = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_c",
                            "type": "function",
                            "function": {
                                "name": "ask_user_for_clarification",
                                "arguments": json.dumps(
                                    {
                                        "question": "请补充角色名。",
                                        "options": ["角色名"],
                                        "reason": "ambiguous",
                                    }
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }
    second = {
        "choices": [{"message": {"role": "assistant", "content": ""}}]
    }
    tools, agent, client = _tool_agent(retrieval_env, [first, second])
    try:
        result = agent.run("有个角色")
    finally:
        agent.close()
        tools.close()
    assert result["clarification"] is not None
    assert result["clarification"]["question"] == "请补充角色名。"
    assert result["answer"] == "请补充角色名。"
    assert result["tool_calls"][0]["tool"] == "ask_user_for_clarification"


def test_llm_agent_invalid_tool_arguments(retrieval_env):
    first = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "type": "function",
                            "function": {
                                "name": "search_subjects",
                                "arguments": "not-json",
                            },
                        }
                    ],
                }
            }
        ]
    }
    second = {
        "choices": [{"message": {"role": "assistant", "content": "没有结果。"}}]
    }
    tools, agent, client = _tool_agent(retrieval_env, [first, second])
    try:
        result = agent.run("查询")
    finally:
        agent.close()
        tools.close()
    assert result["tool_calls"][0]["error"]
    assert result["answer"] == "没有结果。"


def test_llm_agent_max_rounds(retrieval_env):
    tool_call = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_x",
                            "type": "function",
                            "function": {
                                "name": "search_subjects",
                                "arguments": json.dumps({"query": "时间循环"}),
                            },
                        }
                    ],
                }
            }
        ]
    }
    tools, agent, client = _tool_agent(
        retrieval_env, [tool_call, tool_call], config=_config(max_tool_rounds=1)
    )
    try:
        result = agent.run("时间循环")
    finally:
        agent.close()
        tools.close()
    assert result["rounds"] == 1
    assert "max_tool_rounds_reached" in result["warnings"]
    assert len(client.calls) == 2
