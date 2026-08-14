from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context_manager import ContextManagementPolicy, ContextManager
from nanobot.config.schema import ContextManagementConfig
from nanobot.providers.base import LLMResponse
from nanobot.session.artifacts import ToolArtifactStore
from nanobot.session.manager import Session


def _manager(tmp_path, *, estimate=100, window=10_000, recent=2):
    provider = MagicMock()
    provider.estimate_prompt_tokens.return_value = (estimate, "test")
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="## Current Goal and Success Criteria\nkeep marker"))
    policy = ContextManagementPolicy(
        recent_turns=recent, output_reserve_tokens=100, safety_margin_tokens=100,
        tool_offload_threshold_bytes=8192,
    )
    return ContextManager(provider, "test", window, policy, ToolArtifactStore(tmp_path)), provider


@pytest.mark.asyncio
async def test_short_conversation_does_not_compact(tmp_path):
    manager, provider = _manager(tmp_path, estimate=100)
    session = Session(key="test:short", messages=[{"role": "user", "content": "hello"}])
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hello"}]

    prepared = await manager.prepare(messages, [], session=session, session_key=session.key)

    assert prepared.actions == []
    provider.chat_with_retry.assert_not_awaited()
    assert session.context_summary_through == 0


@pytest.mark.asyncio
async def test_long_conversation_compacts_old_turns_and_keeps_recent(tmp_path):
    manager, provider = _manager(tmp_path, estimate=9000, recent=2)
    provider.estimate_prompt_tokens.side_effect = [(9000, "test"), (2000, "test")]
    session = Session(key="test:long")
    for i in range(5):
        session.messages.extend([
            {"role": "user", "content": f"user-{i}"},
            {"role": "assistant", "content": f"assistant-{i}"},
        ])
    messages = [{"role": "system", "content": "system"}, *session.get_context_history(), {"role": "user", "content": "current"}]

    prepared = await manager.prepare(messages, [], session=session, session_key=session.key)

    text = "\n".join(str(m.get("content")) for m in prepared.messages)
    assert "user-0" not in text
    assert "user-3" in text and "user-4" in text and "current" in text
    assert session.context_summary_through == 6
    assert prepared.compacted_turns == 3


@pytest.mark.asyncio
async def test_hard_truncation_removes_complete_tool_turn(tmp_path):
    manager, provider = _manager(tmp_path, estimate=9500, recent=8)
    provider.estimate_prompt_tokens.side_effect = [(9500, "test"), (4000, "test")]
    tool_calls = [
        {"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}},
        {"id": "b", "type": "function", "function": {"name": "y", "arguments": "{}"}},
    ]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "a", "name": "x", "content": "one"},
        {"role": "tool", "tool_call_id": "b", "name": "y", "content": "two"},
        {"role": "user", "content": "current"},
    ]

    prepared = await manager.prepare(messages, [], session_key="test:hard")

    assert prepared.hard_truncated_turns == 1
    assert [m["role"] for m in prepared.messages] == ["system", "user"]
    assert ContextManager.validate_tool_structure(prepared.messages)


def test_validate_tool_structure_rejects_partial_parallel_group(tmp_path):
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "one"},
    ]
    assert not ContextManager.validate_tool_structure(messages)


def test_policy_environment_override(monkeypatch):
    monkeypatch.setenv("NANOBOT_CONTEXT_RECENT_TURNS", "12")
    monkeypatch.setenv("NANOBOT_CONTEXT_TOOL_OFFLOAD_THRESHOLD_BYTES", "99")
    monkeypatch.setenv("NANOBOT_CONTEXT_ARTIFACT_MAX_SEARCHES_PER_SESSION", "4")
    monkeypatch.setenv("NANOBOT_CONTEXT_ARTIFACT_MAX_READS_PER_SESSION", "2")
    monkeypatch.setenv("NANOBOT_CONTEXT_ARTIFACT_MAX_RETURNED_CHARS_PER_SESSION", "8192")
    policy = ContextManagementPolicy.from_config(ContextManagementConfig(), 2048)
    assert policy.recent_turns == 12
    assert policy.tool_offload_threshold_bytes == 99
    assert policy.artifact_max_searches_per_session == 4
    assert policy.artifact_max_reads_per_session == 2
    assert policy.artifact_max_returned_chars_per_session == 8192
    assert policy.output_reserve_tokens == 2048
