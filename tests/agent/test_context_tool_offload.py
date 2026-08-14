import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context_manager import ContextManagementPolicy, ContextManager
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.session.artifacts import ToolArtifactStore


def test_large_tool_result_is_offloaded_and_original_is_recoverable(tmp_path):
    store = ToolArtifactStore(tmp_path)
    policy = ContextManagementPolicy(tool_offload_threshold_bytes=8, tool_summary_max_chars=4)
    manager = ContextManager(MagicMock(), "test", 10000, policy, store)
    messages = [{"role": "tool", "tool_call_id": "call-1", "name": "read", "content": "abcdefghijk"}]

    assert manager.offload_tool_results(messages, "cli:one") == 1
    placeholder = messages[0]["content"]
    artifact_id = placeholder.split("artifact_id: ", 1)[1].splitlines()[0]
    assert "content_type: text/plain" in placeholder
    assert "original_size_bytes: 11" in placeholder
    assert "line_count: 1" in placeholder
    assert "search_tool_result" in placeholder
    assert "get_tool_result" in placeholder
    assert "artifact_path" not in placeholder
    assert "summary: abcd" in placeholder
    assert 'queries=["MARKER", "RESULT", "LARGE-RESULT"]' in placeholder
    assert "answer_ready=true" in placeholder
    assert store.get("cli:one", artifact_id, 0, 100)["content"] == "abcdefghijk"


def test_small_tool_result_stays_inline(tmp_path):
    manager = ContextManager(
        MagicMock(), "test", 10000,
        ContextManagementPolicy(tool_offload_threshold_bytes=20),
        ToolArtifactStore(tmp_path),
    )
    messages = [{"role": "tool", "tool_call_id": "x", "name": "read", "content": "short"}]
    assert manager.offload_tool_results(messages, "cli:one") == 0
    assert messages[0]["content"] == "short"


def test_old_retrieval_results_become_compact_receipts_and_latest_match_stays_inline():
    messages = [
        {"role": "tool", "name": "get_tool_result", "content": json.dumps({
            "artifact_id": "a" * 32, "offset": 0, "end": 4096,
            "content": "x" * 4096,
        })},
        {"role": "tool", "name": "search_tool_result", "content": json.dumps({
            "artifact_id": "a" * 32, "status": "matches", "query": "RESULT",
            "matches": [{"offset": 12007, "snippet": "LARGE-RESULT-993"}],
        })},
        {"role": "tool", "name": "get_tool_result", "content": json.dumps({
            "artifact_id": "a" * 32, "offset": 11800, "end": 12300,
            "content": "LARGE-RESULT-993",
        })},
    ]

    compacted = ContextManager.compact_transient_retrieval_results(messages)

    assert compacted == 2
    assert messages[0]["content"].startswith("[Artifact Retrieval Compacted]")
    assert "range: [0, 4096)" in messages[0]["content"]
    assert "LARGE-RESULT-993" in messages[1]["content"]
    assert "match_offsets: [12007]" not in messages[1]["content"]
    assert messages[2]["content"].startswith("[Artifact Retrieval Compacted]")
    assert "range: [11800, 12300)" in messages[2]["content"]


def test_persisted_retrieval_history_keeps_only_bounded_match_excerpt():
    marker = "LARGE-RESULT-993"
    messages = [{
        "role": "tool", "name": "search_tool_result", "content": json.dumps({
            "artifact_id": "a" * 32,
            "status": "matches",
            "query": "RESULT",
            "matches": [{
                "offset": 500,
                "snippet_start": 0,
                "snippet_end": 1000,
                "snippet": "x" * 500 + marker + "y" * 484,
            }],
        }),
    }]

    assert ContextManager.compact_retrieval_history_for_persistence(messages) == 1
    receipt = messages[0]["content"]
    assert receipt.startswith("[Artifact Retrieval Compacted]")
    assert "match_offsets: [500]" in receipt
    assert marker in receipt
    assert len(receipt) < 600


def test_transient_compaction_strips_old_retrieval_assistant_reasoning():
    messages = [
        {
            "role": "assistant",
            "content": "long analysis",
            "reasoning_content": "r" * 4000,
            "thinking_blocks": [{"text": "hidden"}],
            "tool_calls": [{"id": "call-1", "function": {"name": "get_tool_result"}}],
        },
        {
            "role": "tool", "name": "get_tool_result", "tool_call_id": "call-1",
            "content": json.dumps({
                "artifact_id": "a" * 32, "offset": 0, "end": 4096,
                "content": "x" * 4096,
            }),
        },
        {
            "role": "assistant", "content": "", "reasoning_content": "latest",
            "tool_calls": [{"id": "call-2", "function": {"name": "search_tool_result"}}],
        },
        {
            "role": "tool", "name": "search_tool_result", "tool_call_id": "call-2",
            "content": json.dumps({
                "artifact_id": "a" * 32, "status": "matches", "query": "RESULT",
                "matches": [{"offset": 12007, "snippet": "LARGE-RESULT-993"}],
            }),
        },
    ]

    assert ContextManager.compact_transient_retrieval_results(messages) == 1
    assert messages[0]["content"] == ""
    assert "reasoning_content" not in messages[0]
    assert "thinking_blocks" not in messages[0]
    assert messages[0]["tool_calls"]
    assert messages[2]["reasoning_content"] == "latest"


def test_artifact_gc_deletes_expired_but_protects_active_reference(tmp_path):
    store = ToolArtifactStore(tmp_path)
    expired = store.put("cli:old", "read", "old", "expired")
    protected = store.put("cli:active", "read", "active", "protected")
    old_time = datetime.now() - timedelta(days=60)
    for artifact in (expired, protected):
        meta_path = tmp_path / artifact.relative_path.replace(".data", ".json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["created_at"] = old_time.isoformat()
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = store.collect_garbage(30, {protected.artifact_id})

    assert result == {"deleted": 1, "protected": 1, "invalid": 0}
    assert not (tmp_path / expired.relative_path).exists()
    assert (tmp_path / protected.relative_path).exists()


class _LargeResultTool(Tool):
    @property
    def name(self):
        return "large_result"

    @property
    def description(self):
        return "Return a large fixture."

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return "marker-" + "x" * 9000


@pytest.mark.asyncio
async def test_agent_loop_offloads_before_followup_provider_request(tmp_path):
    provider = MagicMock()
    provider.get_default_model.return_value = "test"
    provider.generation = GenerationSettings(max_tokens=100)
    provider.estimate_prompt_tokens.return_value = (100, "test")
    captured = []

    async def chat(**kwargs):
        captured.append([dict(message) for message in kwargs["messages"]])
        if len(captured) == 1:
            return LLMResponse(content="", tool_calls=[ToolCallRequest("call-1", "large_result", {})])
        return LLMResponse(content="done")

    provider.chat_with_retry = AsyncMock(side_effect=chat)
    loop = AgentLoop(
        bus=MessageBus(), provider=provider, workspace=tmp_path,
        context_window_tokens=20_000,
    )
    loop.tools.register(_LargeResultTool())

    result = await loop.process_direct("run", session_key="cli:offload")

    assert result and result.content == "done"
    second_tool = next(message for message in captured[1] if message.get("role") == "tool")
    assert second_tool["content"].startswith("[Tool Result Offloaded]")
    session = loop.sessions.get_or_create("cli:offload")
    persisted_tool = next(message for message in session.messages if message.get("role") == "tool")
    assert persisted_tool["content"].startswith("[Tool Result Offloaded]")
