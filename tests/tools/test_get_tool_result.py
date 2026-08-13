import asyncio
import json

import pytest

from nanobot.agent.tools.tool_result import GetToolResultTool
from nanobot.session.artifacts import ToolArtifactStore


@pytest.mark.asyncio
async def test_get_tool_result_pages_active_session(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "0123456789")
    tool = GetToolResultTool(store, default_page_size=4)
    tool.set_context("cli:a")

    page = json.loads(await tool.execute(artifact.artifact_id, offset=3, limit=4))
    assert page["content"] == "3456"
    assert page["has_more"] is True


@pytest.mark.asyncio
async def test_get_tool_result_is_session_isolated(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "secret")
    tool = GetToolResultTool(store)
    tool.set_context("cli:b")
    assert (await tool.execute(artifact.artifact_id)).startswith("Error: artifact not found")


@pytest.mark.asyncio
async def test_get_tool_result_context_is_task_local(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact_a = store.put("cli:a", "read", "a", "alpha")
    artifact_b = store.put("cli:b", "read", "b", "beta")
    tool = GetToolResultTool(store)

    async def read(session_key, artifact_id):
        tool.set_context(session_key)
        await asyncio.sleep(0)
        return json.loads(await tool.execute(artifact_id))["content"]

    assert await asyncio.gather(
        read("cli:a", artifact_a.artifact_id),
        read("cli:b", artifact_b.artifact_id),
    ) == ["alpha", "beta"]
