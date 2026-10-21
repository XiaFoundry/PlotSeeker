from __future__ import annotations

import pytest

from agent.agent import PlotSeekerAgent
from agent.tools import TOOL_SPECS, AgentTools
from embeddings.fake import DeterministicFakeEmbedder
from retrieval.keyword_index import KeywordSearcher


@pytest.fixture
def agent_tools(retrieval_env):
    tools = AgentTools(
        keyword_db=retrieval_env["keyword_db"],
        store_dir=retrieval_env["store_dir"],
        checkpoint_path=retrieval_env["checkpoint"],
        sqlite_path=retrieval_env["sqlite"],
        embedder=DeterministicFakeEmbedder(retrieval_env["fake_dimension"]),
    )
    try:
        yield tools
    finally:
        tools.close()


def test_tool_specs_are_explicit(agent_tools):
    specs = agent_tools.specs()
    assert set(specs) == set(TOOL_SPECS)
    for spec in specs.values():
        assert spec["description"]
        assert spec["input_schema"]["type"] == "object"
        assert "required" in spec["input_schema"]


def test_tool_specs_fields():
    assert TOOL_SPECS["search_subjects"]["input_schema"]["required"] == ["query"]
    assert "subject_id" in TOOL_SPECS["get_related_subjects"]["input_schema"][
        "properties"
    ]
    assert "question" in TOOL_SPECS["ask_user_for_clarification"][
        "input_schema"
    ]["properties"]


def test_character_and_episode_tools(agent_tools):
    characters = agent_tools.search_characters("感知时间循环", top_k=5)
    assert characters["count"] > 0
    assert characters["results"][0]["doc_id"] == "character:1"
    assert characters["results"][0]["subject_id"] == 1
    assert characters["results"][0]["relation_path"] == "character:1->subject:1"

    episodes = agent_tools.search_episodes("用脚抵住门", top_k=5)
    assert episodes["count"] > 0
    assert episodes["results"][0]["doc_id"] == "episode:1"
    assert episodes["results"][0]["subject_id"] == 1


def test_related_subjects_tool(agent_tools):
    related = agent_tools.get_related_subjects(1, limit=10)
    assert related["count"] == 2
    ids = {item["subject_id"] for item in related["results"]}
    assert {2, 3} <= ids
    assert all("relation_type" in item for item in related["results"])
    missing = agent_tools.get_related_subjects(99999)
    assert missing["count"] == 0
    assert missing["results"] == []


def test_agent_routes_character_query(agent_tools):
    agent = PlotSeekerAgent(agent_tools, max_steps=4, top_k=5)
    result = agent.run("有个角色能感知时间循环")
    assert result["candidates"]
    assert result["candidates"][0]["subject_id"] == 1
    assert "search_characters" in [call["tool"] for call in result["tool_calls"]]
    assert result["clarification"] is None
    assert result["used_llm"] is False


def test_agent_routes_episode_query(agent_tools):
    agent = PlotSeekerAgent(agent_tools, max_steps=4, top_k=5)
    result = agent.run("记得某一话里，妹妹用脚抵住门")
    assert result["candidates"]
    assert result["candidates"][0]["subject_id"] == 1
    assert "search_episodes" in [call["tool"] for call in result["tool_calls"]]


def test_agent_fetches_related_subjects(agent_tools):
    agent = PlotSeekerAgent(agent_tools, max_steps=4, top_k=5)
    result = agent.run("时间循环学园有什么相关作品")
    assert result["candidates"]
    assert result["related_subjects"]
    assert "get_related_subjects" in [
        call["tool"] for call in result["tool_calls"]
    ]


def test_agent_clarifies_when_unsupported(agent_tools):
    agent = PlotSeekerAgent(
        agent_tools, max_steps=4, top_k=5, clarify_on_low=True
    )
    result = agent.run("zzzz-unfindable-xyz")
    assert result["clarification"] is not None
    assert result["clarification"]["tool"] == "ask_user_for_clarification"
    search_calls = [
        call for call in result["tool_calls"] if call["tool"].startswith("search_")
    ]
    assert len(search_calls) <= 4


def test_agent_evidence_is_traceable(agent_tools, retrieval_env):
    agent = PlotSeekerAgent(agent_tools, max_steps=4, top_k=5)
    result = agent.run("妹妹用脚抵住门")
    searcher = KeywordSearcher(retrieval_env["keyword_db"])
    try:
        for candidate in result["candidates"]:
            assert candidate["subject_id"] in {1, 2, 3}
            for evidence in candidate["evidence"]:
                assert searcher.document(evidence["doc_id"]) is not None
    finally:
        searcher.close()


def test_agent_low_confidence_returns_warning_by_default(agent_tools):
    agent = PlotSeekerAgent(agent_tools, max_steps=4, top_k=5)
    result = agent.run("zzzz-unfindable-xyz")
    assert result["clarification"] is None
    assert result["confidence"] == "low"
    assert "low_confidence_candidates" in result["warnings"]


def test_agent_empty_query_clarifies(agent_tools):
    agent = PlotSeekerAgent(agent_tools, max_steps=2, top_k=5)
    result = agent.run("   ")
    assert result["clarification"] is not None
    assert result["candidates"] == []
    assert result["confidence"] == "none"
