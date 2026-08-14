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
    assert [match["offset"] for match in limited["matches"]] == [2]
    assert [item["offset"] for item in limited["matches"][0]["query_matches"]] == [2, 9, 16]
    assert limited["has_more_matches"] is False


@pytest.mark.asyncio
async def test_search_tool_result_rejects_invalid_and_cross_session_ids(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "secret marker")
    tool = SearchToolResultTool(store, ArtifactRetrievalGuard())
    tool.set_context("cli:b")

    assert (await tool.execute("../artifact", "marker")) == "Error: invalid artifact_id"
    assert await tool.execute("../artifact", queries=["\n"]) == "Error: invalid artifact_id"
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
async def test_answer_ready_requires_a_distinctive_identifier_candidate(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "prefix TARGET suffix")
    guard = ArtifactRetrievalGuard(max_reads=3, max_returned_chars=100)
    getter = GetToolResultTool(store, 8, guard)
    searcher = SearchToolResultTool(store, guard)
    getter.set_context("cli:a")
    searcher.set_context("cli:a")

    match = json.loads(await searcher.execute(artifact.artifact_id, "TARGET", context_chars=2))
    another_search = json.loads(await searcher.execute(artifact.artifact_id, "suffix"))
    away_read = json.loads(await getter.execute(artifact.artifact_id, 0, 4))
    local_read = json.loads(await getter.execute(artifact.artifact_id, 7, 8))

    assert match["answer_ready"] is False
    assert another_search["status"] == "matches"
    assert another_search["answer_ready"] is False
    assert away_read["content"] == "pref"
    assert local_read["content"].startswith("TARGET")

    distinctive = store.put("cli:a", "read", "call-2", "prefix LARGE-RESULT-993 suffix")
    distinctive_result = json.loads(await searcher.execute(
        distinctive.artifact_id, "RESULT", context_chars=20,
    ))
    assert distinctive_result["answer_ready"] is True
    assert distinctive_result["answer_candidates"] == ["LARGE-RESULT-993"]


@pytest.mark.asyncio
async def test_retrieval_ledger_returns_compact_receipts_for_duplicate_search_and_reads(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "0123456789 TARGET suffix")
    guard = ArtifactRetrievalGuard(max_reads=5, max_searches=5, max_returned_chars=100)
    getter = GetToolResultTool(store, 8, guard)
    searcher = SearchToolResultTool(store, guard)
    getter.set_context("cli:a")
    searcher.set_context("cli:a")

    first_search = json.loads(await searcher.execute(
        artifact.artifact_id, "TARGET", context_chars=2,
    ))
    duplicate_search = json.loads(await searcher.execute(
        artifact.artifact_id, "TARGET", context_chars=2,
    ))
    first_read = json.loads(await getter.execute(artifact.artifact_id, 0, 8))
    duplicate_read = json.loads(await getter.execute(artifact.artifact_id, 0, 8))
    overlap_read = json.loads(await getter.execute(artifact.artifact_id, 4, 8))

    assert first_search["status"] == "matches"
    assert duplicate_search["status"] == "cached_receipt"
    assert duplicate_search["match_offsets"] == [11]
    assert first_read["content"] == "01234567"
    assert duplicate_read["status"] == "cached_receipt"
    assert overlap_read["status"] == "overlap_already_retrieved"
    assert duplicate_read["overlapping_ranges"][0]["source"] == "get"


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


@pytest.mark.asyncio
async def test_multi_query_search_loads_once_merges_and_attributes(tmp_path, monkeypatch):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put(
        "cli:a", "read", "call",
        "prefix MARKER: LARGE-RESULT-993 suffix far-away RESULT",
    )
    original = store._load_text
    loads = 0

    def counted_load(session_key, artifact_id):
        nonlocal loads
        loads += 1
        return original(session_key, artifact_id)

    monkeypatch.setattr(store, "_load_text", counted_load)
    tool = SearchToolResultTool(store, ArtifactRetrievalGuard(), total_snippet_chars=60)
    tool.set_context("cli:a")
    result = json.loads(await tool.execute(
        artifact.artifact_id,
        queries=["MARKER", "RESULT", "MARKER", "missing"],
        context_chars=8,
    ))

    assert loads == 1
    assert result["queries"] == ["MARKER", "RESULT", "missing"]
    assert result["answer_ready"] is True
    assert result["answer_candidates"] == ["LARGE-RESULT-993"]
    assert {item["query"] for item in result["matches"][0]["query_matches"]} >= {
        "MARKER", "RESULT",
    }
    assert sum(len(item["snippet"]) for item in result["matches"]) <= 60


@pytest.mark.asyncio
async def test_multi_query_search_handles_distinct_regions_and_no_matches(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "alpha" + "x" * 40 + "beta")
    tool = SearchToolResultTool(store, ArtifactRetrievalGuard())
    tool.set_context("cli:a")

    result = json.loads(await tool.execute(
        artifact.artifact_id, queries=["alpha", "beta"], context_chars=2,
    ))
    missing = json.loads(await tool.execute(
        artifact.artifact_id, queries=["gamma", "delta"], context_chars=2,
    ))

    assert len(result["matches"]) == 2
    assert [match["queries"] for match in result["matches"]] == [["alpha"], ["beta"]]
    assert missing["status"] == "no_match"

    limited = json.loads(await tool.execute(
        artifact.artifact_id, queries=["alpha", "beta"], max_matches=1, context_chars=2,
    ))
    assert len(limited["matches"]) == 1
    assert limited["has_more_matches"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_query", ["", "   ", "\n", ":", "?", "A", "7"])
async def test_query_guard_rejects_low_information_queries(tmp_path, bad_query):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "TARGET 中文 词")
    tool = SearchToolResultTool(store, ArtifactRetrievalGuard())
    tool.set_context("cli:a")

    result = json.loads(await tool.execute(artifact.artifact_id, queries=[bad_query]))

    assert result["status"] == "query_too_broad"
    assert result["rejected_queries"]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["词", "中文", "go", "TARGET"])
async def test_query_guard_allows_short_and_utf8_queries(tmp_path, query):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "词 中文 go TARGET")
    tool = SearchToolResultTool(store, ArtifactRetrievalGuard())
    tool.set_context("cli:a")

    result = json.loads(await tool.execute(artifact.artifact_id, queries=[query]))

    assert result["status"] == "matches"
    assert result["rejected_queries"] == []


@pytest.mark.asyncio
async def test_multi_query_drops_invalid_items_but_searches_valid_ones(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "prefix TARGET suffix")
    tool = SearchToolResultTool(store, ArtifactRetrievalGuard())
    tool.set_context("cli:a")

    result = json.loads(await tool.execute(
        artifact.artifact_id, queries=["\n", "TARGET", "?"], context_chars=1,
    ))

    assert result["status"] == "matches"
    assert result["queries"] == ["TARGET"]
    assert len(result["rejected_queries"]) == 2


@pytest.mark.asyncio
async def test_answer_ready_converges_search_and_allows_one_local_get(tmp_path):
    store = ToolArtifactStore(tmp_path)
    artifact = store.put("cli:a", "read", "call", "prefix LARGE-RESULT-993 suffix")
    guard = ArtifactRetrievalGuard(max_reads=5, max_searches=5, max_returned_chars=500)
    searcher = SearchToolResultTool(store, guard)
    getter = GetToolResultTool(store, 12, guard)
    searcher.set_context("cli:a")
    getter.set_context("cli:a")

    hit = json.loads(await searcher.execute(artifact.artifact_id, query="RESULT"))
    repeated = json.loads(await searcher.execute(artifact.artifact_id, query="prefix"))
    unrelated = json.loads(await getter.execute(artifact.artifact_id, offset=0, limit=3))
    local = json.loads(await getter.execute(
        artifact.artifact_id, offset=hit["matches"][0]["offset"], limit=12,
    ))
    second_local = json.loads(await getter.execute(
        artifact.artifact_id, offset=hit["matches"][0]["offset"], limit=12,
    ))

    assert hit["answer_ready"] is True
    assert repeated["status"] == "answer_ready_receipt"
    assert unrelated["status"] == "answer_ready_receipt"
    assert "RESULT-993" in local["content"]
    assert second_local["status"] == "answer_ready_receipt"


def test_search_tool_descriptions_prefer_multi_query_without_banning_known_zero(tmp_path):
    store = ToolArtifactStore(tmp_path)
    guard = ArtifactRetrievalGuard()
    search_description = SearchToolResultTool(store, guard).description
    get_description = GetToolResultTool(store, guard=guard).description

    assert "up to five queries" in search_description
    assert "answer_ready" in search_description
    assert "Offset 0 is valid" in get_description
    assert "blind scanning" in get_description
