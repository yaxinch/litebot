from pathlib import Path

from benchmarks.providers.scripted import ScriptedProvider
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.session.manager import Session, SessionManager


def _loop(tmp_path: Path, mode: str) -> AgentLoop:
    return AgentLoop(
        bus=MessageBus(), provider=ScriptedProvider([{"content": "ok"}]),
        workspace=tmp_path, benchmark_context_mode=mode,
    )


def test_baseline_uses_full_history_without_summary(tmp_path):
    loop = _loop(tmp_path, "baseline")
    session = Session(
        key="test:baseline", context_summary="summary", context_summary_through=2,
        messages=[{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"},
                  {"role": "user", "content": "recent"}],
    )
    history, summary = loop._session_context(session)
    assert [item["content"] for item in history] == ["old", "answer", "recent"]
    assert summary is None
    assert loop._context_management_enabled is False


def test_context_mode_uses_summary_view(tmp_path):
    loop = _loop(tmp_path, "context_management")
    session = Session(
        key="test:new", context_summary="summary", context_summary_through=2,
        messages=[{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"},
                  {"role": "user", "content": "recent"}],
    )
    history, summary = loop._session_context(session)
    assert [item["content"] for item in history] == ["recent"]
    assert summary == "summary"
    assert loop._context_management_enabled is True


def test_baseline_restores_legacy_tool_result_truncation(tmp_path):
    loop = _loop(tmp_path, "baseline")
    session = Session(key="test:save")
    content = "x" * (loop._TOOL_RESULT_MAX_CHARS + 10)
    start = {"role": "user", "content": "start"}
    loop._save_turn(session, [start, {"role": "tool", "tool_call_id": "1", "name": "fixture", "content": content}], start_message=start)
    assert session.messages[-1]["content"].endswith("... (truncated)")
    assert session.messages[-1]["content"].startswith("x" * loop._TOOL_RESULT_MAX_CHARS)
    assert "x" * (loop._TOOL_RESULT_MAX_CHARS + 1) not in session.messages[-1]["content"]


async def test_baseline_never_offloads_during_agent_iterations(tmp_path):
    provider = ScriptedProvider([{"content": "ok"}])
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("test:offload")
    session.add_message("user", "old")
    session.add_message("assistant", "stored")
    sessions.save(session)
    loop = AgentLoop(
        bus=MessageBus(), provider=provider, workspace=tmp_path,
        session_manager=sessions, benchmark_context_mode="baseline",
    )
    await loop.process_direct("new", session_key=session.key)
    assert loop.context_manager.total_offloaded_artifacts == 0
    assert loop.context_manager.telemetry == []
