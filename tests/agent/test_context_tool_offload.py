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
    assert "artifact_path: sessions/artifacts/cli_one/" in placeholder
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
