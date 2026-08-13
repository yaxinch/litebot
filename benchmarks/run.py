from __future__ import annotations

import argparse
import asyncio
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import AsyncExitStack, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.collector import (
    BenchmarkCollector,
    RecordingProvider,
    RecordingTool,
    add_usage,
)
from benchmarks.compare import compare_runs
from benchmarks.models import BenchmarkCase, load_cases
from benchmarks.providers.scripted import ScriptedProvider
from benchmarks.tools.fixtures import EchoTool, ErrorTool, ExplodingTool, LookupTool
from nanobot.agent.context import ContextBuilder
from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory import MemoryStore
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.mcp import connect_mcp_servers
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import MCPServerConfig
from nanobot.session.manager import SessionManager

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = Path(__file__).with_name("cases")
RESULTS_ROOT = ROOT / "benchmark_results"
EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"id": call_id, "name": name, "arguments": arguments}


def _script_for(case_id: str) -> list[dict[str, Any]]:
    final = {
        "det_long_context_direct_recall": "ORCHID-7429",
        "det_long_context_tail_instruction": "TAIL-OK",
        "det_long_context_history_boundary": "boundary valid",
        "det_history_same_session": ["Stored Aurora-17.", "Aurora-17"],
        "det_history_after_reload": "RELOAD-31",
        "det_history_tool_result": "TOOL-HISTORY-9",
        "det_memory_long_term_recall": "ARCHIVE-88",
    }.get(case_id)
    if isinstance(final, list):
        return [{"content": item} for item in final]
    if isinstance(final, str):
        return [{"content": final}]

    scripts: dict[str, list[dict[str, Any]]] = {
        "det_long_context_consolidation": [
            {
                "content": "",
                "tool_calls": [_call("mem1", "save_memory", {
                    "history_entry": "[2026-08-12 00:00] Archived ARCHIVE-88",
                    "memory_update": "Preferred codename: ARCHIVE-88",
                })],
            },
            {"content": "ARCHIVE-88"},
        ],
        "det_multi_tool_sequential": [
            {"content": "", "tool_calls": [_call("s1", "lookup", {"key": "alpha"})]},
            {"content": "", "tool_calls": [_call("s2", "echo", {"value": "A-17"})]},
            {"content": "A-17"},
        ],
        "det_multi_tool_parallel": [
            {"content": "", "tool_calls": [_call("p1", "lookup", {"key": "alpha"}), _call("p2", "lookup", {"key": "beta"})]},
            {"content": "A-17 B-29"},
        ],
        "det_multi_tool_result_synthesis": [
            {"content": "", "tool_calls": [_call("m1", "lookup", {"key": "alpha"}), _call("m2", "lookup", {"key": "beta"}), _call("m3", "lookup", {"key": "gamma"})]},
            {"content": "A-17 B-29 G-41"},
        ],
        "det_multi_round_tool_chain": [
            {"content": "", "tool_calls": [_call("c1", "lookup", {"key": "alpha"})]},
            {"content": "", "tool_calls": [_call("c2", "lookup", {"key": "beta"})]},
            {"content": "", "tool_calls": [_call("c3", "lookup", {"key": "gamma"})]},
            {"content": "chain complete"},
        ],
        "det_tool_invalid_parameters_recovery": [
            {"content": "", "tool_calls": [_call("v1", "echo", {})]},
            {"content": "", "tool_calls": [_call("v2", "echo", {"value": "recovered"})]},
            {"content": "recovered"},
        ],
        "det_tool_error_result_recovery": [
            {"content": "", "tool_calls": [_call("e1", "error_tool", {"reason": "planned"})]},
            {"content": "", "tool_calls": [_call("e2", "echo", {"value": "fallback-ok"})]},
            {"content": "fallback-ok"},
        ],
        "det_tool_exception_recovery": [
            {"content": "", "tool_calls": [_call("x1", "exploding_tool", {})]},
            {"content": "exception handled"},
        ],
        "det_duplicate_tool_same_round": [
            {"content": "", "tool_calls": [_call("d1", "echo", {"value": "same"}), _call("d2", "echo", {"value": "same"})]},
            {"content": "done"},
        ],
        "det_duplicate_tool_across_rounds": [
            {"content": "", "tool_calls": [_call("a1", "echo", {"value": "same"})]},
            {"content": "", "tool_calls": [_call("a2", "echo", {"value": "same"})]},
            {"content": "done"},
        ],
        "det_repeated_failure_until_limit": [
            {"content": "", "tool_calls": [_call(f"r{i}", "error_tool", {"reason": "repeat"})]} for i in range(3)
        ],
        "int_real_filesystem_chain": [
            {"content": "", "tool_calls": [_call("f1", "write_file", {"path": "artifact.txt", "content": "draft"})]},
            {"content": "", "tool_calls": [_call("f2", "read_file", {"path": "artifact.txt"})]},
            {"content": "", "tool_calls": [_call("f3", "edit_file", {"path": "artifact.txt", "old_text": "draft", "new_text": "edited"})]},
            {"content": "", "tool_calls": [_call("f4", "read_file", {"path": "artifact.txt"})]},
            {"content": "edited"},
        ],
        "int_real_shell_chain": [
            {"content": "", "tool_calls": [_call("sh1", "exec", {"command": "echo shell-ok > shell-artifact.txt"})]},
            {"content": "", "tool_calls": [_call("sh2", "read_file", {"path": "shell-artifact.txt"})]},
            {"content": "shell-ok"},
        ],
        "int_local_mcp_roundtrip": [
            {"content": "", "tool_calls": [_call("mc1", "mcp_fixture_echo", {"value": "mcp-ok"})]},
            {"content": "mcp-ok"},
        ],
        "int_subagent_roundtrip": [
            {"content": "", "tool_calls": [_call("sp1", "spawn", {"task": "Reply exactly subagent-ok"})]},
            {"content": "subagent-ok"},
            {"content": "subagent-ok"},
        ],
    }
    return scripts[case_id]


def _git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except Exception:
        return "unknown", True


def _blocked_external_io(*args: Any, **kwargs: Any):
    raise RuntimeError("deterministic benchmark attempted forbidden external I/O")


def _base_registry(workspace: Path, collector: BenchmarkCollector) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (EchoTool(), LookupTool(), ErrorTool(), ExplodingTool()):
        registry.register(RecordingTool(tool, collector))
    return registry


def _add_integration_tools(case: BenchmarkCase, workspace: Path, registry: ToolRegistry, collector: BenchmarkCollector) -> None:
    if case.id == "int_real_filesystem_chain":
        tools = [WriteFileTool(workspace, workspace), ReadFileTool(workspace, workspace), EditFileTool(workspace, workspace)]
    elif case.id == "int_real_shell_chain":
        tools = [ExecTool(working_dir=str(workspace), restrict_to_workspace=True), ReadFileTool(workspace, workspace)]
    else:
        tools = []
    for tool in tools:
        registry.register(RecordingTool(tool, collector))


def _seed_workspace(case: BenchmarkCase, workspace: Path) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    memory_dir = workspace / "memory"
    sessions_dir = workspace / "sessions"
    if "memory" in case.fixtures:
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / "MEMORY.md").write_text("Preferred codename: ARCHIVE-88", encoding="utf-8")
    if "session_reload" in case.fixtures:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        history.append({"role": "user", "content": "Reloaded fact is RELOAD-31."})
    if "tool_history" in case.fixtures:
        history.extend([
            {"role": "user", "content": "Look up the prior value."},
            {"role": "assistant", "content": None, "tool_calls": [_call("old1", "lookup", {"key": "old"})]},
            {"role": "tool", "tool_call_id": "old1", "name": "lookup", "content": "TOOL-HISTORY-9"},
        ])
    if "history_boundary" in case.fixtures:
        history.extend([{"role": "user", "content": "boundary start"}, {"role": "assistant", "content": "boundary valid"}])
    if "long_context" in case.fixtures:
        marker = (Path(__file__).with_name("fixtures") / "long_context" / "context.txt").read_text(encoding="utf-8")
        history.extend([{"role": "user", "content": ("offline filler " * 4000) + marker}, {"role": "assistant", "content": "Context received."}])
    return history


def _result_text(messages: list[dict[str, Any]]) -> str:
    return json.dumps(messages, ensure_ascii=False)


def _assert_case(case: BenchmarkCase, result: dict[str, Any], workspace: Path) -> list[dict[str, Any]]:
    output, events = result.get("final_output") or "", result.get("tool_events", [])
    names = [item["name"] for item in events]
    session_text = result.get("session_text", "")
    memory_path = workspace / "memory" / "MEMORY.md"
    memory_text = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
    checked = []
    for assertion in case.assertions:
        kind, expected = assertion.type, assertion.value
        actual: Any = None
        if kind == "output_contains":
            actual, passed = output, str(expected) in output
        elif kind == "output_not_contains":
            actual, passed = output, str(expected) not in output
        elif kind == "output_regex":
            actual, passed = output, re.search(str(expected), output) is not None
        elif kind == "output_equals":
            actual, passed = output, output == expected
        elif kind == "tool_called":
            actual, passed = names, expected in names
        elif kind == "tool_not_called":
            actual, passed = names, expected not in names
        elif kind.startswith("tool_calls_"):
            actual = result["tool_calls"]
            passed = _numeric(kind, actual, expected)
        elif kind.startswith("tool_rounds_"):
            actual = result["tool_rounds"]
            passed = _numeric(kind, actual, expected)
        elif kind == "tool_error_count":
            actual, passed = result["tool_errors"], result["tool_errors"] == expected
        elif kind == "stop_reason":
            actual, passed = result["stop_reason"], result["stop_reason"] == expected
        elif kind == "session_contains":
            actual, passed = session_text, str(expected) in session_text
        elif kind == "memory_contains":
            actual, passed = memory_text, str(expected) in memory_text
        elif kind in {"file_exists", "file_contains"}:
            target = (workspace / (assertion.path or str(expected))).resolve()
            if workspace.resolve() not in target.parents and target != workspace.resolve():
                actual, passed = str(target), False
            elif kind == "file_exists":
                actual, passed = target.exists(), target.exists()
            else:
                actual = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
                passed = str(expected) in actual
        else:
            actual, passed = None, False
        checked.append({"type": kind, "expected": expected, "actual": actual if isinstance(actual, (int, float, bool, list)) else str(actual)[:500], "passed": passed})
    return checked


def _numeric(kind: str, actual: int, expected: int) -> bool:
    if kind.endswith("_exact"):
        return actual == expected
    if kind.endswith("_min"):
        return actual >= expected
    if kind.endswith("_max"):
        return actual <= expected
    return False


async def _run_standard(case: BenchmarkCase, workspace: Path, provider, collector: BenchmarkCollector, registry: ToolRegistry) -> dict[str, Any]:
    runner = AgentRunner(provider)
    history = _seed_workspace(case, workspace)
    sessions = SessionManager(workspace)
    session = sessions.get_or_create("benchmark:case")
    session.messages = [{**item, "timestamp": datetime.now(timezone.utc).isoformat()} for item in history]
    sessions.save(session)
    if "session_reload" in case.fixtures:
        sessions.invalidate(session.key)
        session = sessions.get_or_create(session.key)
    context = ContextBuilder(workspace, timezone="UTC")
    final = None
    stop_reason = "completed"
    runner_events: list[dict[str, Any]] = []
    for turn in case.turns:
        history = session.get_history(max_messages=0)
        messages = context.build_messages(history=history, current_message=turn.content, channel="benchmark", chat_id="case")
        run = await runner.run(AgentRunSpec(initial_messages=messages, tools=registry, model=provider.get_default_model(), max_iterations=case.max_iterations, concurrent_tools=True))
        final, stop_reason = run.final_content, run.stop_reason
        session.add_message("user", turn.content)
        for item in run.messages[len(messages):]:
            session.messages.append({**item, "timestamp": datetime.now(timezone.utc).isoformat()})
        sessions.save(session)
        runner_events.extend(run.tool_events)
    # Registry validation errors occur before RecordingTool.execute; retain them as synthetic events.
    recorded_error_counts: dict[str, int] = {}
    for event in collector.tool_events:
        if event.get("status") != "ok":
            name = str(event["name"])
            recorded_error_counts[name] = recorded_error_counts.get(name, 0) + 1
    for event in runner_events:
        if event.get("status") != "error":
            continue
        name = str(event["name"])
        if recorded_error_counts.get(name, 0):
            recorded_error_counts[name] -= 1
            continue
        collector.tool_events.append({
            "name": name, "status": "error_result",
            "detail": event.get("detail", ""), "latency_ms": 0,
        })
    sessions.invalidate(session.key)
    persisted = sessions.get_or_create(session.key)
    return {"final_output": final or "", "stop_reason": stop_reason, "session_text": _result_text(persisted.messages)}


async def _run_agent_loop_case(case: BenchmarkCase, workspace: Path, provider, collector: BenchmarkCollector, registry: ToolRegistry) -> dict[str, Any]:
    """Exercise the complete context -> loop -> session persistence path."""
    sessions = SessionManager(workspace)
    session = sessions.get_or_create("benchmark:case")
    session.messages = [
        {**item, "timestamp": datetime.now(timezone.utc).isoformat()}
        for item in _seed_workspace(case, workspace)
    ]
    sessions.save(session)
    if "session_reload" in case.fixtures:
        sessions.invalidate(session.key)
    loop = AgentLoop(
        bus=MessageBus(), provider=provider, workspace=workspace,
        model=provider.get_default_model(), max_iterations=case.max_iterations,
        context_window_tokens=1_000_000, restrict_to_workspace=True,
        session_manager=sessions,
    )
    loop.tools = registry
    final = ""
    try:
        for turn in case.turns:
            outbound = await loop.process_direct(turn.content, session_key=session.key, channel="benchmark", chat_id="case")
            final = outbound.content if outbound else ""
        await loop.close_mcp()
    finally:
        loop.stop()
    sessions.invalidate(session.key)
    persisted = sessions.get_or_create(session.key)
    return {"final_output": final, "stop_reason": "completed", "session_text": _result_text(persisted.messages)}


async def _run_consolidation(case: BenchmarkCase, workspace: Path, provider: RecordingProvider, collector: BenchmarkCollector, registry: ToolRegistry) -> dict[str, Any]:
    provider.phase = "memory"
    store = MemoryStore(workspace)
    ok = await store.consolidate(
        [{"role": "user", "content": "Archive marker ARCHIVE-88", "timestamp": "2026-08-12T00:00:00"}],
        provider,
        provider.get_default_model(),
    )
    if not ok:
        raise RuntimeError("deterministic memory consolidation failed")
    provider.phase = "agent"
    return await _run_standard(case, workspace, provider, collector, registry)


async def _run_mcp(case: BenchmarkCase, workspace: Path, provider, collector: BenchmarkCollector, registry: ToolRegistry) -> dict[str, Any]:
    server = Path(__file__).with_name("fixtures") / "mcp" / "server.py"
    cfg = MCPServerConfig(type="stdio", command=sys.executable, args=[str(server)], enabled_tools=["*"])
    async with AsyncExitStack() as stack:
        await connect_mcp_servers({"fixture": cfg}, registry, stack)
        tool = registry.get("mcp_fixture_echo")
        if tool:
            registry.unregister(tool.name)
            registry.register(RecordingTool(tool, collector))
        return await _run_standard(case, workspace, provider, collector, registry)


async def _run_mcp_and_subagent(case: BenchmarkCase, workspace: Path, provider, collector: BenchmarkCollector, registry: ToolRegistry) -> dict[str, Any]:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.agent.tools.spawn import SpawnTool

    server = Path(__file__).with_name("fixtures") / "mcp" / "server.py"
    cfg = MCPServerConfig(type="stdio", command=sys.executable, args=[str(server)], enabled_tools=["*"])
    bus = MessageBus()
    manager = SubagentManager(provider=provider, workspace=workspace, bus=bus, model=provider.get_default_model(), restrict_to_workspace=True)
    spawn = SpawnTool(manager)
    spawn.set_context("cli", "benchmark")
    registry.register(RecordingTool(spawn, collector))
    async with AsyncExitStack() as stack:
        await connect_mcp_servers({"fixture": cfg}, registry, stack)
        tool = registry.get("mcp_fixture_echo")
        if tool:
            registry.unregister(tool.name)
            registry.register(RecordingTool(tool, collector))
        outcome = await _run_standard(case, workspace, provider, collector, registry)
        deadline = time.monotonic() + 60
        while manager.get_running_count() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if manager.get_running_count():
            raise TimeoutError("live subagent did not finish")
        return outcome


async def _run_subagent(case: BenchmarkCase, workspace: Path, provider, collector: BenchmarkCollector, registry: ToolRegistry) -> dict[str, Any]:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.agent.tools.spawn import SpawnTool
    bus = MessageBus()
    manager = SubagentManager(provider=provider, workspace=workspace, bus=bus, model=provider.get_default_model(), restrict_to_workspace=True)
    spawn = SpawnTool(manager)
    spawn.set_context("cli", "benchmark")
    registry.register(RecordingTool(spawn, collector))
    outcome = await _run_standard(case, workspace, provider, collector, registry)
    deadline = time.monotonic() + 10
    while manager.get_running_count() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    if manager.get_running_count():
        await manager.cancel_by_session("cli:benchmark")
        raise TimeoutError("subagent fixture did not finish")
    if bus.inbound_size:
        message = await bus.consume_inbound()
        outcome["final_output"] = message.content
    return outcome


def _load_live_provider():
    from nanobot.cli.commands import _make_provider
    from nanobot.config.loader import load_config
    config = load_config()
    try:
        provider = _make_provider(config)
    except BaseException as exc:
        raise ConnectionError("configured live provider is unavailable") from exc
    return provider, config


async def _run_live(case: BenchmarkCase, workspace: Path, collector: BenchmarkCollector) -> tuple[dict[str, Any], str, str]:
    provider, config = _load_live_provider()
    recorded = RecordingProvider(provider, collector)
    registry = _base_registry(workspace, collector)
    if case.id == "live_web_search":
        from nanobot.agent.tools.web import WebSearchTool
        registry.register(RecordingTool(WebSearchTool(config.tools.web.search), collector))
    if case.id == "live_web_fetch_and_shell":
        from nanobot.agent.tools.web import WebFetchTool
        registry.register(RecordingTool(WebFetchTool(), collector))
        registry.register(RecordingTool(ExecTool(working_dir=str(workspace), restrict_to_workspace=True), collector))
    if case.id == "live_mcp_and_subagent":
        outcome = await _run_mcp_and_subagent(case, workspace, recorded, collector, registry)
        return outcome, provider.get_default_model(), config.get_provider_name()
    outcome = await _run_standard(case, workspace, recorded, collector, registry)
    return outcome, provider.get_default_model(), config.get_provider_name()


async def run_case(case: BenchmarkCase, run_id: str, artifacts_root: Path, allow_live: bool) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    commit, dirty = _git_info()
    collector = BenchmarkCollector()
    base = {
        "schema_version": "litebot-benchmark-result/v1", "run_id": run_id,
        "case_id": case.id, "suite": case.suite, "status": "FAILED", "skip_reason": None,
        "baseline_ref": "v0.1.4.post6", "git_commit": commit, "git_dirty": dirty,
        "model": "scripted-v1", "provider": "scripted", "started_at": started_at,
    }
    workspace = Path(tempfile.mkdtemp(prefix=f"litebot-bench-{case.id}-"))
    artifact_dir = artifacts_root / case.id
    try:
        if case.suite == "live" and not allow_live:
            raise PermissionError("live suite requires --allow-live")
        registry = _base_registry(workspace, collector)
        if case.suite == "integration":
            _add_integration_tools(case, workspace, registry, collector)
        if case.suite == "live":
            outcome, model, provider_name = await _run_live(case, workspace, collector)
            base.update(model=model, provider=provider_name or "configured")
        else:
            scripted = ScriptedProvider(_script_for(case.id))
            recorded = RecordingProvider(scripted, collector)
            if case.suite == "deterministic":
                from unittest.mock import patch

                guard = (
                    patch("socket.getaddrinfo", _blocked_external_io),
                    patch("socket.create_connection", _blocked_external_io),
                    patch("httpx.AsyncClient.request", _blocked_external_io),
                    patch("asyncio.create_subprocess_exec", _blocked_external_io),
                    patch("asyncio.create_subprocess_shell", _blocked_external_io),
                )
            else:
                guard = (nullcontext(),)
            with guard[0], guard[1] if len(guard) > 1 else nullcontext(), guard[2] if len(guard) > 2 else nullcontext(), guard[3] if len(guard) > 3 else nullcontext(), guard[4] if len(guard) > 4 else nullcontext():
                if case.id == "det_long_context_consolidation":
                    outcome = await _run_consolidation(case, workspace, recorded, collector, registry)
                elif case.category in {"long_context", "history_recall"}:
                    outcome = await _run_agent_loop_case(case, workspace, recorded, collector, registry)
                elif case.id == "int_local_mcp_roundtrip":
                    outcome = await _run_mcp(case, workspace, recorded, collector, registry)
                elif case.id == "int_subagent_roundtrip":
                    outcome = await _run_subagent(case, workspace, recorded, collector, registry)
                else:
                    outcome = await _run_standard(case, workspace, recorded, collector, registry)
        base.update(outcome)
        base["tool_events"] = collector.tool_events
        base["iterations"] = sum(item.get("phase", "agent") == "agent" for item in collector.rounds)
        base["tool_rounds"] = collector.tool_rounds
        base["tool_calls"] = collector.tool_calls
        base["tool_errors"] = collector.tool_errors
        checks = _assert_case(case, base, workspace)
        base["assertions"] = checks
        base["status"] = "PASSED" if all(item["passed"] for item in checks) else "FAILED"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for path in workspace.rglob("*"):
            if path.is_file() and path.stat().st_size <= 1_000_000:
                target = artifact_dir / path.relative_to(workspace)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    except PermissionError as exc:
        base.update(status="SKIPPED", skip_reason=str(exc), final_output="", stop_reason="skipped", assertions=[])
    except (FileNotFoundError, ConnectionError) as exc:
        base.update(status="SKIPPED" if case.suite != "deterministic" else "FAILED", skip_reason=str(exc), final_output="", stop_reason="error", assertions=[], error=f"{type(exc).__name__}: {exc}")
    except BaseException as exc:
        message = f"{type(exc).__name__}: {exc}"
        environment_error = case.suite == "live" and any(marker in message.lower() for marker in ("api key", "connect", "network", "resolve", "timeout", "rate limit", "quota"))
        base.update(status="SKIPPED" if environment_error else "FAILED", skip_reason=message if environment_error else None, final_output="", stop_reason="error", assertions=[], error=message)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    agent_usage, memory_usage = dict(EMPTY_USAGE), dict(EMPTY_USAGE)
    for round_item in collector.rounds:
        if round_item.get("phase") == "memory":
            memory_usage = add_usage(memory_usage, round_item["usage"])
        else:
            agent_usage = add_usage(agent_usage, round_item["usage"])
    base["usage"] = {"agent": agent_usage, "memory": memory_usage, "combined": add_usage(agent_usage, memory_usage)}
    base["rounds"] = collector.rounds
    base["tool_events"] = collector.tool_events
    base["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
    base.setdefault("error", None)
    return base


def _summary(run_id: str, suite: str, results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    counts = {status: sum(item["status"] == status for item in results) for status in ("PASSED", "FAILED", "SKIPPED")}
    usage = dict(EMPTY_USAGE)
    for item in results:
        usage = add_usage(usage, item["usage"]["combined"])
    commit, dirty = _git_info()
    executed = counts["PASSED"] + counts["FAILED"]
    return {
        "schema_version": "litebot-benchmark-summary/v1", "run_id": run_id, "suite": suite,
        "counts": counts, "coverage": 0 if not results else round(executed / len(results), 4),
        "usage": usage, "duration_ms": round(sum(item["duration_ms"] for item in results), 3),
        "tool_calls": sum(item["tool_calls"] for item in results), "tool_errors": sum(item["tool_errors"] for item in results),
        "git_commit": commit, "git_dirty": dirty, "python": platform.python_version(), "os": platform.platform(),
        "models": sorted({str(item.get("model", "unknown")) for item in results}),
        "providers": sorted({str(item.get("provider", "unknown")) for item in results}),
        "arguments": vars(args),
    }


async def run_suite(args: argparse.Namespace) -> int:
    cases = load_cases(CASES_DIR / f"{args.suite}.json", args.suite)
    if args.case:
        cases = [case for case in cases if case.id == args.case]
        if not cases:
            print(f"Unknown case for suite {args.suite}: {args.case}", file=sys.stderr)
            return 2
    if args.suite == "live" and not args.allow_live:
        print("Live benchmark is disabled by default; pass --allow-live.", file=sys.stderr)
        return 2
    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.suite}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(args.output or RESULTS_ROOT / run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = run_dir / "artifacts"
    results: list[dict[str, Any]] = []
    with (run_dir / "results.jsonl").open("a", encoding="utf-8") as stream:
        for case in cases:
            result = await run_case(case, run_id, artifacts, args.allow_live)
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{result['status']:7} {case.id} ({result['duration_ms']:.1f} ms)")
    summary = _summary(run_id, args.suite, results, args)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results: {run_dir}")
    return 1 if summary["counts"]["FAILED"] else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LiteBot baseline benchmark")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run")
    run.add_argument("--suite", choices=("deterministic", "integration", "live"), default="deterministic")
    run.add_argument("--case")
    run.add_argument("--allow-live", action="store_true")
    run.add_argument("--output")
    run.add_argument("--run-id")
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        values = ["run", "--suite", "deterministic"]
    args = _parser().parse_args(values)
    if args.command == "compare":
        report, failed = compare_runs(Path(args.baseline), Path(args.candidate))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(failed)
    if args.command == "run":
        return asyncio.run(run_suite(args))
    _parser().print_help()
    return 2
