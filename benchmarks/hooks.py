"""Deterministic end-to-end benchmark for the lifecycle hook framework."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from benchmarks.providers.scripted import ScriptedProvider
from benchmarks.tools.fixtures import EchoTool, ErrorTool, LookupTool
from nanobot.agent.hook import (
    AgentHook,
    AgentHookContext,
    HookDecision,
    HookManager,
    LifecycleEvent,
    LifecycleEventType,
)
from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus

SCHEMA = "litebot-hook-benchmark/v1"
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "benchmark_results"
PUBLIC_EVENTS = (
    LifecycleEventType.SESSION_START,
    LifecycleEventType.USER_PROMPT_SUBMIT,
    LifecycleEventType.AGENT_RUN_START,
    LifecycleEventType.PRE_LLM_CALL,
    LifecycleEventType.POST_LLM_CALL,
    LifecycleEventType.PRE_TOOL_USE,
    LifecycleEventType.POST_TOOL_USE,
    LifecycleEventType.TOOL_ERROR,
    LifecycleEventType.AGENT_RUN_END,
    LifecycleEventType.SESSION_END,
)


@dataclass(slots=True)
class Check:
    name: str
    category: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class CaseOutcome:
    case_id: str
    checks: list[Check]
    trace: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    missing_events: int = 0
    unexpected_events: int = 0
    duplicate_events: int = 0

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


class RecordingHook:
    """Benchmark-only lifecycle recorder with stable JSON-safe fields."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def register(self, manager: HookManager, *, priority: int = -1000) -> None:
        manager.register(PUBLIC_EVENTS, self, priority=priority, name="benchmark-recorder")

    def __call__(self, event: LifecycleEvent) -> None:
        tool_call = event.payload.get("tool_call")
        record = {
            "event": event.type.value,
            "phase": event.phase,
            "session_id": event.session_key,
            "run_id": event.run_id,
            "iteration": event.iteration,
            "channel": event.channel,
            "model": event.payload.get("model"),
            "provider": event.payload.get("provider"),
            "tool_call_id": getattr(tool_call, "id", event.payload.get("tool_call_id")),
            "tool_name": getattr(tool_call, "name", event.payload.get("name")),
            "tool_arguments": getattr(tool_call, "arguments", event.payload.get("arguments")),
            "result": event.payload.get("result"),
            "error": event.payload.get("error"),
            "failure_count": len(event.failures),
        }
        if isinstance(record["result"], AgentRunResult):
            record["result"] = record["result"].final_content
        self.records.append(record)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(LookupTool())
    registry.register(ErrorTool())
    return registry


async def _run(
    script: list[dict[str, Any]],
    *,
    manager: HookManager | None = None,
    hook: AgentHook | None = None,
    session: str = "benchmark:hooks",
    concurrent_tools: bool = False,
) -> tuple[AgentRunResult, ScriptedProvider]:
    provider = ScriptedProvider(script)
    result = await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "fixed input"}],
            tools=_registry(),
            model=provider.get_default_model(),
            max_iterations=8,
            hook=hook,
            hook_manager=manager,
            session_key=session,
            event_metadata={"channel": "benchmark", "chat_id": session},
            concurrent_tools=concurrent_tools,
        )
    )
    return result, provider


def _types(trace: list[dict[str, Any]]) -> list[str]:
    return [record["event"] for record in trace]


def _sequence_check(actual: list[str], expected: list[str]) -> tuple[Check, int, int, int]:
    actual_counts, expected_counts = Counter(actual), Counter(expected)
    missing = sum((expected_counts - actual_counts).values())
    unexpected = sum((actual_counts - expected_counts).values())
    duplicate = sum(max(0, actual_counts[key] - expected_counts[key]) for key in actual_counts)
    return (
        Check(
            "exact_event_sequence",
            "event_order",
            actual == expected,
            f"actual={actual!r}, expected={expected!r}",
        ),
        missing,
        unexpected,
        duplicate,
    )


def _outcome(
    case_id: str,
    checks: list[Check],
    recorder: RecordingHook,
    expected: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
) -> CaseOutcome:
    missing = unexpected = duplicate = 0
    if expected is not None:
        check, missing, unexpected, duplicate = _sequence_check(_types(recorder.records), expected)
        checks.insert(0, check)
    return CaseOutcome(
        case_id, checks, recorder.records, metrics or {}, missing, unexpected, duplicate
    )


async def session_lifecycle() -> CaseOutcome:
    manager, recorder = HookManager(), RecordingHook()
    recorder.register(manager)
    provider = ScriptedProvider([{"content": "one"}, {"content": "two"}])
    with tempfile.TemporaryDirectory(prefix="hook-session-") as root:
        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=Path(root),
            context_window_tokens=1_000_000,
        )
        loop.hooks = manager
        loop.context_manager.hook_manager = manager
        loop.memory_consolidator.store.hook_manager = manager
        await loop.process_direct("turn one", session_key="benchmark:same", channel="benchmark")
        await loop.process_direct("turn two", session_key="benchmark:same", channel="benchmark")
        await loop.close_mcp()
    types = _types(recorder.records)
    return _outcome(
        "session_lifecycle",
        [
            Check("session_start_once", "lifecycle", types.count("SessionStart") == 1),
            Check("session_end_once", "lifecycle", types.count("SessionEnd") == 1),
            Check(
                "run_pair_per_turn",
                "lifecycle",
                types.count("AgentRunStart") == types.count("AgentRunEnd") == 2,
            ),
        ],
        recorder,
    )


async def single_run_order() -> CaseOutcome:
    manager, recorder = HookManager(), RecordingHook()
    recorder.register(manager)
    result, _ = await _run([{"content": "done"}], manager=manager)
    expected = ["AgentRunStart", "PreLLMCall", "PostLLMCall", "AgentRunEnd"]
    return _outcome(
        "single_run_order",
        [Check("task_success", "task_success", result.final_content == "done")],
        recorder,
        expected,
    )


async def tool_run_order() -> CaseOutcome:
    manager, recorder = HookManager(), RecordingHook()
    recorder.register(manager)
    result, _ = await _run(
        [
            {
                "content": "",
                "tool_calls": [{"id": "c1", "name": "echo", "arguments": {"value": "ok"}}],
            },
            {"content": "final"},
        ],
        manager=manager,
    )
    expected = [
        "AgentRunStart",
        "PreLLMCall",
        "PostLLMCall",
        "PreToolUse",
        "PostToolUse",
        "PreLLMCall",
        "PostLLMCall",
        "AgentRunEnd",
    ]
    return _outcome(
        "tool_run_order",
        [Check("task_success", "task_success", result.final_content == "final")],
        recorder,
        expected,
    )


async def multi_tool_order() -> CaseOutcome:
    manager, recorder = HookManager(), RecordingHook()
    recorder.register(manager)
    result, _ = await _run(
        [
            {
                "content": "",
                "tool_calls": [
                    {"id": "a", "name": "echo", "arguments": {"value": "A"}},
                    {"id": "b", "name": "lookup", "arguments": {"key": "beta"}},
                ],
            },
            {"content": "final"},
        ],
        manager=manager,
    )
    expected = [
        "AgentRunStart",
        "PreLLMCall",
        "PostLLMCall",
        "PreToolUse",
        "PostToolUse",
        "PreToolUse",
        "PostToolUse",
        "PreLLMCall",
        "PostLLMCall",
        "AgentRunEnd",
    ]
    tools = [
        record["tool_name"]
        for record in recorder.records
        if record["event"] in {"PreToolUse", "PostToolUse"}
    ]
    return _outcome(
        "multi_tool_order",
        [
            Check("task_success", "task_success", result.final_content == "final"),
            Check(
                "tool_pair_order",
                "payload",
                tools == ["echo", "echo", "lookup", "lookup"],
                repr(tools),
            ),
        ],
        recorder,
        expected,
    )


async def multi_round_run() -> CaseOutcome:
    manager, recorder = HookManager(), RecordingHook()
    recorder.register(manager)
    result, _ = await _run(
        [
            {"tool_calls": [{"id": "a", "name": "echo", "arguments": {"value": "A"}}]},
            {"tool_calls": [{"id": "b", "name": "lookup", "arguments": {"key": "alpha"}}]},
            {"content": "final"},
        ],
        manager=manager,
    )
    types = _types(recorder.records)
    return _outcome(
        "multi_round_run",
        [
            Check("task_success", "task_success", result.final_content == "final"),
            Check(
                "single_run_boundary",
                "lifecycle",
                types.count("AgentRunStart") == types.count("AgentRunEnd") == 1,
            ),
            Check(
                "llm_count",
                "lifecycle",
                types.count("PreLLMCall") == types.count("PostLLMCall") == 3,
            ),
            Check(
                "tool_count",
                "lifecycle",
                types.count("PreToolUse") == types.count("PostToolUse") == 2,
            ),
        ],
        recorder,
    )


async def cross_turn_session() -> CaseOutcome:
    outcome = await session_lifecycle()
    outcome.case_id = "cross_turn_session"
    run_ids = [record["run_id"] for record in outcome.trace if record["event"] == "AgentRunStart"]
    session_ids = {record["session_id"] for record in outcome.trace if record["session_id"]}
    outcome.checks.extend(
        [
            Check(
                "distinct_run_ids", "payload", len(run_ids) == len(set(run_ids)) == 2, repr(run_ids)
            ),
            Check(
                "stable_session_id", "payload", session_ids == {"benchmark:same"}, repr(session_ids)
            ),
        ]
    )
    return outcome


async def hook_payload() -> CaseOutcome:
    manager, recorder = HookManager(), RecordingHook()
    recorder.register(manager)
    result, _ = await _run(
        [
            {
                "tool_calls": [
                    {"id": "ok1", "name": "echo", "arguments": {"value": "payload"}},
                    {"id": "err1", "name": "error_tool", "arguments": {"reason": "expected"}},
                ]
            },
            {"content": "done"},
        ],
        manager=manager,
        session="benchmark:payload",
    )
    starts = [record for record in recorder.records if record["event"] == "AgentRunStart"]
    pre_tools = [record for record in recorder.records if record["event"] == "PreToolUse"]
    post_tools = [record for record in recorder.records if record["event"] == "PostToolUse"]
    errors = [record for record in recorder.records if record["event"] == "ToolError"]
    all_run_ids = {record["run_id"] for record in recorder.records if record["run_id"]}
    return _outcome(
        "hook_payload",
        [
            Check("task_success", "task_success", result.final_content == "done"),
            Check(
                "run_metadata",
                "payload",
                len(starts) == 1
                and starts[0]["model"] == "scripted-v1"
                and starts[0]["provider"] == "ScriptedProvider",
            ),
            Check(
                "stable_identifiers",
                "payload",
                all_run_ids == {result.run_id}
                and all(record["session_id"] == "benchmark:payload" for record in recorder.records),
            ),
            Check(
                "tool_arguments",
                "payload",
                [item["tool_arguments"] for item in pre_tools]
                == [{"value": "payload"}, {"reason": "expected"}],
            ),
            Check(
                "tool_result",
                "payload",
                len(post_tools) == 1 and post_tools[0]["result"] == "payload",
            ),
            Check(
                "tool_error", "payload", len(errors) == 1 and errors[0]["tool_name"] == "error_tool"
            ),
        ],
        recorder,
    )


async def multiple_hooks_order() -> CaseOutcome:
    manager, recorder = HookManager(), RecordingHook()
    calls: list[str] = []
    manager.register(LifecycleEventType.PRE_LLM_CALL, lambda event: calls.append("low"), priority=0)
    manager.register(
        LifecycleEventType.PRE_LLM_CALL, lambda event: calls.append("high-a"), priority=10
    )
    manager.register(
        LifecycleEventType.PRE_LLM_CALL, lambda event: calls.append("high-b"), priority=10
    )
    recorder.register(manager)
    await _run([{"content": "done"}], manager=manager)
    return _outcome(
        "multiple_hooks_order",
        [
            Check(
                "priority_then_registration",
                "event_order",
                calls == ["high-a", "high-b", "low"],
                repr(calls),
            ),
        ],
        recorder,
    )


async def before_hook_failure() -> CaseOutcome:
    manager, recorder = HookManager(), RecordingHook()

    def broken(event: LifecycleEvent) -> None:
        raise RuntimeError("before failure")

    manager.register(LifecycleEventType.PRE_LLM_CALL, broken, priority=10, name="broken-before")
    recorder.register(manager)
    logger.disable("nanobot.agent.hook")
    try:
        result, provider = await _run([{"content": "done"}], manager=manager)
    finally:
        logger.enable("nanobot.agent.hook")
    pre = next(record for record in recorder.records if record["event"] == "PreLLMCall")
    return _outcome(
        "before_hook_failure",
        [
            Check("task_success", "task_success", result.final_content == "done"),
            Check("failure_recorded", "failure_policy", pre["failure_count"] == 1),
            Check("operation_continued", "failure_policy", provider.index == 1),
        ],
        recorder,
    )


async def after_hook_failure() -> CaseOutcome:
    manager, recorder = HookManager(), RecordingHook()

    def broken(event: LifecycleEvent) -> None:
        raise RuntimeError("after failure")

    manager.register(LifecycleEventType.POST_LLM_CALL, broken, priority=10, name="broken-after")
    recorder.register(manager)
    logger.disable("nanobot.agent.hook")
    try:
        result, provider = await _run([{"content": "original"}], manager=manager)
    finally:
        logger.enable("nanobot.agent.hook")
    post = next(record for record in recorder.records if record["event"] == "PostLLMCall")
    return _outcome(
        "after_hook_failure",
        [
            Check(
                "original_result_preserved", "failure_policy", result.final_content == "original"
            ),
            Check("failure_recorded", "failure_policy", post["failure_count"] == 1),
            Check("no_extra_llm_call", "failure_policy", provider.index == 1),
        ],
        recorder,
    )


async def legacy_adapter() -> CaseOutcome:
    class LegacyRecorder(AgentHook):
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def before_iteration(self, context: AgentHookContext) -> None:
            self.calls.append("before")

        async def after_iteration(self, context: AgentHookContext) -> None:
            self.calls.append("after")

        def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
            self.calls.append("finalize")
            return content.upper() if content else content

    direct_manager, direct_recorder = HookManager(), RecordingHook()
    direct_recorder.register(direct_manager)
    direct, _ = await _run([{"content": "done"}], manager=direct_manager)
    legacy_manager, legacy_events, legacy = HookManager(), RecordingHook(), LegacyRecorder()
    legacy_events.register(legacy_manager)
    adapted, _ = await _run([{"content": "done"}], manager=legacy_manager, hook=legacy)
    direct_public = _types(direct_recorder.records)
    adapted_public = _types(legacy_events.records)
    return _outcome(
        "legacy_adapter",
        [
            Check("legacy_result", "compatibility", adapted.final_content == "DONE"),
            Check(
                "legacy_method_order",
                "compatibility",
                legacy.calls == ["before", "finalize", "after"],
                repr(legacy.calls),
            ),
            Check(
                "public_lifecycle_equivalent",
                "compatibility",
                direct_public == adapted_public,
                f"direct={direct_public}, adapted={adapted_public}",
            ),
            Check("direct_result_unchanged", "task_success", direct.final_content == "done"),
        ],
        legacy_events,
    )


async def hook_state_isolation() -> CaseOutcome:
    manager, recorder = HookManager(), RecordingHook()
    recorder.register(manager)
    observed: list[tuple[str | None, dict[str, Any]]] = []

    def stateful(event: LifecycleEvent) -> None:
        event.state["owner"] = event.session_key
        observed.append((event.session_key, dict(event.state)))

    manager.register(LifecycleEventType.AGENT_RUN_START, stateful, priority=10)
    first, _ = await _run([{"content": "a"}], manager=manager, session="benchmark:A")
    second, _ = await _run([{"content": "b"}], manager=manager, session="benchmark:B")
    return _outcome(
        "hook_state_isolation",
        [
            Check("distinct_run_ids", "payload", first.run_id != second.run_id),
            Check(
                "isolated_state",
                "payload",
                observed
                == [
                    (
                        "benchmark:A",
                        {"channel": "benchmark", "chat_id": "benchmark:A", "owner": "benchmark:A"},
                    ),
                    (
                        "benchmark:B",
                        {"channel": "benchmark", "chat_id": "benchmark:B", "owner": "benchmark:B"},
                    ),
                ],
                repr(observed),
            ),
        ],
        recorder,
    )


async def decision_control() -> CaseOutcome:
    modify_manager = HookManager()
    modify_manager.register(
        LifecycleEventType.POST_LLM_CALL, lambda event: HookDecision.modify({"content": "modified"})
    )
    modified, modified_provider = await _run([{"content": "original"}], manager=modify_manager)
    deny_manager = HookManager()
    deny_manager.register(
        LifecycleEventType.PRE_LLM_CALL, lambda event: HookDecision.deny("blocked", "denied")
    )
    denied, denied_provider = await _run([{"content": "must not run"}], manager=deny_manager)
    recorder = RecordingHook()
    return _outcome(
        "decision_control",
        [
            Check("modify_applied", "lifecycle", modified.final_content == "modified"),
            Check("modify_no_extra_call", "lifecycle", modified_provider.index == 1),
            Check(
                "deny_short_circuit",
                "lifecycle",
                denied.stop_reason == "denied" and denied.final_content == "denied",
            ),
            Check("denied_operation_not_called", "lifecycle", denied_provider.index == 0),
        ],
        recorder,
    )


async def hook_overhead(repetitions: int = 30) -> CaseOutcome:
    baseline_times: list[float] = []
    hooked_times: list[float] = []
    baseline_calls = hooked_calls = baseline_tokens = hooked_tokens = 0
    for _ in range(max(3, repetitions)):
        started = time.perf_counter()
        baseline, provider = await _run([{"content": "done"}])
        baseline_times.append((time.perf_counter() - started) * 1000)
        baseline_calls += provider.index
        baseline_tokens += sum(baseline.usage.values())

        manager, recorder = HookManager(), RecordingHook()
        recorder.register(manager)
        started = time.perf_counter()
        hooked, provider = await _run([{"content": "done"}], manager=manager)
        hooked_times.append((time.perf_counter() - started) * 1000)
        hooked_calls += provider.index
        hooked_tokens += sum(hooked.usage.values())
    base_median = statistics.median(baseline_times)
    hook_median = statistics.median(hooked_times)
    delta = hook_median - base_median
    metrics = {
        "baseline_median_ms": round(base_median, 4),
        "hook_median_ms": round(hook_median, 4),
        "overhead_ms": round(delta, 4),
        "overhead_ratio": round(hook_median / base_median, 4) if base_median else None,
    }
    return _outcome(
        "hook_overhead",
        [
            Check("same_llm_round_trips", "task_success", baseline_calls == hooked_calls),
            Check("same_token_usage", "task_success", baseline_tokens == hooked_tokens),
            Check("bounded_absolute_overhead", "task_success", delta < 20.0, repr(metrics)),
        ],
        RecordingHook(),
        metrics=metrics,
    )


CASES: dict[str, Callable[..., Awaitable[CaseOutcome]]] = {
    "session_lifecycle": session_lifecycle,
    "single_run_order": single_run_order,
    "tool_run_order": tool_run_order,
    "multi_tool_order": multi_tool_order,
    "multi_round_run": multi_round_run,
    "cross_turn_session": cross_turn_session,
    "hook_payload": hook_payload,
    "multiple_hooks_order": multiple_hooks_order,
    "before_hook_failure": before_hook_failure,
    "after_hook_failure": after_hook_failure,
    "legacy_adapter": legacy_adapter,
    "hook_state_isolation": hook_state_isolation,
    "decision_control": decision_control,
    "hook_overhead": hook_overhead,
}


async def run_benchmark(*, case: str | None = None, repetitions: int = 30) -> dict[str, Any]:
    selected = [case] if case else list(CASES)
    if any(name not in CASES for name in selected):
        raise ValueError(f"unknown hook benchmark case: {case}")
    outcomes: list[CaseOutcome] = []
    for name in selected:
        started = time.perf_counter()
        try:
            outcome = await (CASES[name](repetitions) if name == "hook_overhead" else CASES[name]())
        except BaseException as exc:
            outcome = CaseOutcome(
                name,
                [Check("case_execution", "task_success", False, f"{type(exc).__name__}: {exc}")],
            )
        outcome.metrics["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        outcomes.append(outcome)
    category_map = {
        "task_success": "task_success",
        "lifecycle": "lifecycle_correct",
        "event_order": "event_order_correct",
        "payload": "payload_correct",
        "failure_policy": "failure_policy_correct",
        "compatibility": "compatibility_correct",
    }
    rates: dict[str, float] = {}
    for source, target in category_map.items():
        checks = [
            check for outcome in outcomes for check in outcome.checks if check.category == source
        ]
        rates[target] = 1.0 if not checks else sum(check.passed for check in checks) / len(checks)
    performance = next(
        (outcome.metrics for outcome in outcomes if outcome.case_id == "hook_overhead"), {}
    )
    return {
        "schema_version": SCHEMA,
        "created_at": datetime.now().isoformat(),
        "python": platform.python_version(),
        "cases": len(outcomes),
        "passed": sum(outcome.passed for outcome in outcomes),
        "failed": sum(not outcome.passed for outcome in outcomes),
        **rates,
        "unexpected_events": sum(outcome.unexpected_events for outcome in outcomes),
        "missing_events": sum(outcome.missing_events for outcome in outcomes),
        "duplicate_events": sum(outcome.duplicate_events for outcome in outcomes),
        "performance": performance,
        "results": [
            {**asdict(outcome), "status": "PASSED" if outcome.passed else "FAILED"}
            for outcome in outcomes
        ],
    }


def _print_summary(report: dict[str, Any]) -> None:
    for result in report["results"]:
        print(
            f"{result['status']:7} {result['case_id']} ({result['metrics']['duration_ms']:.1f} ms)"
        )
    print("\nHook Benchmark Summary\n")
    print(f"Cases:                    {report['cases']}")
    print(f"Passed:                   {report['passed']}")
    print(f"Failed:                   {report['failed']}")
    for key, label in (
        ("task_success", "Task Success"),
        ("lifecycle_correct", "Lifecycle Correct"),
        ("event_order_correct", "Event Order Correct"),
        ("payload_correct", "Payload Correct"),
        ("failure_policy_correct", "Failure Policy Correct"),
        ("compatibility_correct", "Compatibility Correct"),
    ):
        print(f"{label + ':':26} {report[key] * 100:.1f}%")
    print(f"\nUnexpected Events:        {report['unexpected_events']}")
    print(f"Missing Events:           {report['missing_events']}")
    print(f"Duplicate Events:         {report['duplicate_events']}")
    performance = report.get("performance") or {}
    if performance:
        print(f"\nBaseline median latency: {performance['baseline_median_ms']:.4f} ms")
        print(f"Hook-enabled median:      {performance['hook_median_ms']:.4f} ms")
        print(f"Hook overhead:            {performance['overhead_ms']:.4f} ms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiteBot deterministic Hook lifecycle benchmark")
    parser.add_argument("--case", choices=tuple(CASES))
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = asyncio.run(run_benchmark(case=args.case, repetitions=max(3, args.repetitions)))
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-hooks-{uuid.uuid4().hex[:8]}"
    output = Path(args.output) if args.output else RESULTS_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for result in report["results"]:
            stream.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    summary = {key: value for key, value in report.items() if key != "results"}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_summary(report)
    print(f"Results: {output}")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
