import asyncio
import json

import pytest

from nanobot.agent.tools.tool_result import (
    ArtifactRetrievalGuard,
    GetToolResultTool,
    SearchToolResultTool,
)
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "query", "expected_offset"),
    [
        ("MARKER tail", "MARKER", 0),
        ("head MARKER tail", "MARKER", 5),
        ("head MARKER", "MARKER", 5),
        ("前文 中文标记 后文", "中文标记", 3),
    ],
)
async def test_search_tool_result_finds_utf8_marker_positions(
    tmp_path, text, query, expected_offset,
):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", text)
    guard = ArtifactRetrievalGuard()
    tool = SearchToolResultTool(store, guard)
    tool.set_context("cli:a")

    result = json.loads(await tool.execute(
        artifact.artifact_id, query, context_chars=2,
    ))

    assert result["status"] == "matches"
    assert result["matches"][0]["offset"] == expected_offset
    assert query in result["matches"][0]["snippet"]


@pytest.mark.asyncio
async def test_search_tool_result_no_match_and_multiple_match_limit(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "x MARK x MARK x MARK")
    tool = SearchToolResultTool(store, ArtifactRetrievalGuard())
    tool.set_context("cli:a")

    missing = json.loads(await tool.execute(artifact.artifact_id, "absent"))
    limited = json.loads(await tool.execute(artifact.artifact_id, "MARK", max_matches=2))

    assert missing["status"] == "no_match"
    assert missing["matches"] == []
    assert [match["offset"] for match in limited["matches"]] == [2, 9]
    assert limited["has_more_matches"] is True


@pytest.mark.asyncio
async def test_search_tool_result_rejects_invalid_and_cross_session_ids(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "secret marker")
    tool = SearchToolResultTool(store, ArtifactRetrievalGuard())
    tool.set_context("cli:b")

    assert (await tool.execute("../artifact", "marker")) == "Error: invalid artifact_id"
    assert (await tool.execute(artifact.artifact_id, "marker")).startswith(
        "Error: artifact not found"
    )


@pytest.mark.asyncio
async def test_get_tool_result_caps_page_and_blocks_after_three_reads(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "0123456789abcdef")
    tool = GetToolResultTool(store, default_page_size=4, guard=ArtifactRetrievalGuard(
        max_reads=3, max_returned_chars=12, max_sequential_reads=3,
    ))
    tool.set_context("cli:a")

    pages = [json.loads(await tool.execute(artifact.artifact_id, offset=i * 4, limit=99)) for i in range(3)]
    blocked = await tool.execute(artifact.artifact_id, offset=12, limit=4)

    assert [len(page["content"]) for page in pages] == [4, 4, 4]
    assert "retrieval guard blocked" in blocked
    assert "search_tool_result" in blocked


@pytest.mark.asyncio
async def test_get_tool_result_enforces_total_char_and_sequential_guards(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "0123456789abcdef")
    char_guard_tool = GetToolResultTool(store, 4, ArtifactRetrievalGuard(
        max_reads=10, max_returned_chars=6, max_sequential_reads=10,
    ))
    char_guard_tool.set_context("cli:a")
    first = json.loads(await char_guard_tool.execute(artifact.artifact_id, 0, 4))
    second = json.loads(await char_guard_tool.execute(artifact.artifact_id, 8, 4))
    blocked_chars = await char_guard_tool.execute(artifact.artifact_id, 12, 4)

    sequential_tool = GetToolResultTool(store, 2, ArtifactRetrievalGuard(
        max_reads=10, max_returned_chars=100, max_sequential_reads=2,
    ))
    sequential_tool.set_context("cli:a")
    await sequential_tool.execute(artifact.artifact_id, 0, 2)
    await sequential_tool.execute(artifact.artifact_id, 2, 2)
    blocked_sequential = await sequential_tool.execute(artifact.artifact_id, 4, 2)

    assert len(first["content"]) == 4
    assert len(second["content"]) == 2
    assert "maximum returned artifact characters" in blocked_chars
    assert "sequential offset scan detected" in blocked_sequential


@pytest.mark.asyncio
async def test_successful_search_resets_get_budget_for_local_read(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "prefix TARGET suffix")
    guard = ArtifactRetrievalGuard(max_reads=1, max_returned_chars=100)
    getter = GetToolResultTool(store, 6, guard)
    searcher = SearchToolResultTool(store, guard)
    getter.set_context("cli:a")
    searcher.set_context("cli:a")

    await getter.execute(artifact.artifact_id, 0, 6)
    assert "retrieval guard blocked" in await getter.execute(artifact.artifact_id, 6, 6)
    match = json.loads(await searcher.execute(artifact.artifact_id, "TARGET"))
    local = json.loads(await getter.execute(
        artifact.artifact_id, match["matches"][0]["offset"], 6,
    ))

    assert local["content"] == "TARGET"


@pytest.mark.asyncio
async def test_search_tool_result_guard_limits_calls_and_returned_chars(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "prefix TARGET suffix")
    guarded = SearchToolResultTool(store, ArtifactRetrievalGuard(
        max_searches=1, max_returned_chars=100,
    ))
    guarded.set_context("cli:a")
    assert json.loads(await guarded.execute(artifact.artifact_id, "absent"))["status"] == "no_match"
    assert "maximum searches reached" in await guarded.execute(artifact.artifact_id, "TARGET")

    char_guarded = SearchToolResultTool(store, ArtifactRetrievalGuard(
        max_searches=5, max_returned_chars=3,
    ))
    char_guarded.set_context("cli:a")
    blocked = await char_guarded.execute(
        artifact.artifact_id, "TARGET", context_chars=0,
    )
    assert "exceed maximum returned artifact characters" in blocked
