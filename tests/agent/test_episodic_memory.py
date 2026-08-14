from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.context_manager import ContextManagementPolicy, ContextManager
from nanobot.agent.episodic_memory import (
    EPISODIC_MEMORY_HEADING,
    EpisodicMemorySource,
    EpisodicMemoryStore,
    format_retrieval_context,
)
from nanobot.agent.memory import MemoryStore
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.session.artifacts import ToolArtifactStore
from nanobot.session.manager import Session


def _entry(
    content: str,
    *,
    timestamp: str = "2026-08-14T00:00:00+00:00",
    category: str = "fact",
    importance: int = 3,
) -> dict:
    return {
        "timestamp": timestamp,
        "category": category,
        "content": content,
        "importance": importance,
    }


def test_structured_append_writes_jsonl_and_markdown_mirror(tmp_path: Path) -> None:
    store = EpisodicMemoryStore(tmp_path)
    source = EpisodicMemorySource(
        session_key="telegram:123", message_start=4, message_end=9,
    )

    added = store.append([_entry("Use PostgreSQL for the project.", category="decision", importance=4)], source=source)

    assert len(added) == 1
    raw = json.loads(store.jsonl_file.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["source"]["session_key"] == "telegram:123"
    assert raw["importance"] == 4
    markdown = store.history_file.read_text(encoding="utf-8")
    assert f"<!-- episodic:id={added[0].id} -->" in markdown
    assert "Use PostgreSQL" in markdown


def test_exact_duplicate_is_suppressed_at_write_time(tmp_path: Path) -> None:
    store = EpisodicMemoryStore(tmp_path)
    assert len(store.append([_entry("Stable constraint")])) == 1
    assert store.append([_entry("  stable   CONSTRAINT  ", importance=5)]) == ()
    assert len(store.load_entries()) == 1


def test_legacy_history_migration_is_non_destructive_and_idempotent(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    original = (
        "[2026-01-02 03:04] User selected the blue deployment.\n\n"
        "[2026-01-03 04:05] [RAW] 1 messages\n[2026-01-03] USER: raw marker\n\n"
        '{"timestamp":"2026-01-04","summary":"JSON legacy marker","importance":4}\n\n'
    )
    history = memory_dir / "HISTORY.md"
    history.write_text(original, encoding="utf-8")
    store = EpisodicMemoryStore(tmp_path)

    first = store.load_entries()
    store._cache_key = None
    second = store.load_entries()

    assert history.read_text(encoding="utf-8") == original
    assert len(first) == len(second) == 3
    assert {item.category for item in first} == {"legacy", "raw_archive"}
    assert len(store.jsonl_file.read_text(encoding="utf-8").splitlines()) == 3


def test_invalid_jsonl_line_is_skipped_without_blocking_retrieval(tmp_path: Path) -> None:
    store = EpisodicMemoryStore(tmp_path)
    store.jsonl_file.write_text("not-json\n", encoding="utf-8")
    store.append([_entry("Recoverable marker ALPHA-42")])

    result = store.retrieve("ALPHA-42")

    assert len(result.entries) == 1
    assert "ALPHA-42" in result.entries[0].entry.content


def test_retrieval_supports_english_and_cjk_tokens(tmp_path: Path) -> None:
    store = EpisodicMemoryStore(tmp_path)
    store.append([
        _entry("Deployment uses Kubernetes namespace litebot-prod", category="project_state"),
        _entry("项目约束：第一版不要引入外部向量数据库", category="constraint", importance=5),
        _entry("Unrelated lunch discussion"),
    ])

    english = store.retrieve("Which Kubernetes namespace is used?", top_k=1)
    chinese = store.retrieve("向量数据库有什么项目约束？", top_k=1)

    assert "litebot-prod" in english.entries[0].entry.content
    assert "不要引入" in chinese.entries[0].entry.content
    assert english.entries[0].lexical_score == pytest.approx(1.0)


def test_score_combines_lexical_recency_and_importance(tmp_path: Path) -> None:
    store = EpisodicMemoryStore(tmp_path)
    now = datetime(2026, 8, 14, tzinfo=timezone.utc)
    store.append([
        _entry("release marker recent detail", timestamp=(now - timedelta(days=1)).isoformat(), importance=1),
        _entry("release marker older detail", timestamp=(now - timedelta(days=90)).isoformat(), importance=5),
    ])

    result = store.retrieve("release marker", now=now)

    assert len(result.entries) == 2
    recent, older = result.entries
    assert recent.entry.content.endswith("recent detail")
    assert recent.recency_score > older.recency_score
    assert recent.importance_score < older.importance_score
    assert recent.relevance_score == pytest.approx(
        0.70 * recent.lexical_score + 0.20 * recent.recency_score + 0.10 * recent.importance_score
    )


def test_near_duplicate_and_character_budget_suppression(tmp_path: Path) -> None:
    store = EpisodicMemoryStore(tmp_path)
    store.append([
        _entry("alpha beta gamma delta epsilon", category="decision", importance=5),
        _entry("alpha beta gamma delta epsilon.", category="decision", importance=4),
        _entry("alpha independent short", category="fact"),
    ])

    result = store.retrieve("alpha beta gamma", top_k=5, char_budget=500)

    assert result.skipped_duplicates == 1
    assert len(result.entries) == 2
    assert result.injected_chars <= 500


def test_context_builder_places_ephemeral_memory_before_summary(tmp_path: Path) -> None:
    store = EpisodicMemoryStore(tmp_path)
    store.append([_entry("Relevant marker EPS-7")])
    result = store.retrieve("EPS-7")
    builder = ContextBuilder(tmp_path)

    messages = builder.build_messages(
        history=[{"role": "user", "content": "old turn"}],
        current_message="EPS-7?",
        session_summary="summary marker",
        episodic_memory=result,
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["content"].startswith(EPISODIC_MEMORY_HEADING)
    assert messages[1]["content"] == format_retrieval_context(result)
    assert len(messages[1]["content"]) == result.injected_chars
    assert "summary marker" in messages[2]["content"]
    assert messages[3]["content"] == "old turn"


@pytest.mark.asyncio
async def test_context_manager_protects_episodic_system_message_during_compaction(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.estimate_prompt_tokens.side_effect = [(9000, "test"), (2000, "test")]
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="summary"))
    manager = ContextManager(
        provider,
        "test",
        10_000,
        ContextManagementPolicy(
            recent_turns=1,
            output_reserve_tokens=100,
            safety_margin_tokens=100,
            tool_offload_threshold_bytes=8192,
        ),
        ToolArtifactStore(tmp_path),
    )
    session = Session(key="cli:test")
    for index in range(3):
        session.messages.extend([
            {"role": "user", "content": f"user-{index}"},
            {"role": "assistant", "content": f"assistant-{index}"},
        ])
    episodic_store = EpisodicMemoryStore(tmp_path)
    episodic_store.append([_entry("Protected episodic marker")])
    retrieval = episodic_store.retrieve("episodic marker")
    messages = ContextBuilder(tmp_path).build_messages(
        history=session.get_context_history(),
        current_message="current",
        episodic_memory=retrieval,
    )

    prepared = await manager.prepare(
        messages, [], session=session, session_key=session.key, episodic_memory=retrieval,
    )

    assert any(
        str(message.get("content", "")).startswith(EPISODIC_MEMORY_HEADING)
        for message in prepared.messages
    )
    assert prepared.episodic_entries == 1
    assert manager.telemetry[-1]["episodic_candidates"] == 1


@pytest.mark.asyncio
async def test_consolidation_accepts_structured_entries_and_updates_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            id="save",
            name="save_memory",
            arguments={
                "history_entries": [
                    _entry("Selected JSONL sidecar", category="decision", importance=5),
                    _entry("No vector database", category="constraint", importance=5),
                ],
                "memory_update": "# Memory\nStable user fact",
            },
        )],
    ))

    ok = await store.consolidate(
        [{"role": "user", "content": "plan", "timestamp": "2026-08-14T00:00:00"}],
        provider,
        "test",
        source=EpisodicMemorySource(session_key="cli:test", message_start=0, message_end=1),
    )

    assert ok
    assert len(store.episodic.load_entries()) == 2
    assert "Stable user fact" in store.memory_file.read_text(encoding="utf-8")
    assert "Selected JSONL sidecar" in store.history_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_agent_loop_retrieves_per_turn_without_persisting_memory_block(tmp_path: Path) -> None:
    from nanobot.agent.loop import AgentLoop

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=100)
    provider.estimate_prompt_tokens.return_value = (100, "test")
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="ok"))
    provider.chat_stream_with_retry = provider.chat_with_retry
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        context_window_tokens=10_000,
    )
    loop.context.memory.episodic.append([_entry("Loop marker EPS-LOOP-9")])

    outbound = await loop.process_direct("What is EPS-LOOP-9?", session_key="cli:test")
    await loop.close_mcp()

    assert outbound is not None and outbound.content == "ok"
    assert loop.context_manager.telemetry[0]["episodic_entries"] == 1
    persisted_text = "\n".join(
        str(message.get("content", ""))
        for message in loop.sessions.get_or_create("cli:test").messages
    )
    assert EPISODIC_MEMORY_HEADING not in persisted_text
    assert "What is EPS-LOOP-9?" in persisted_text
