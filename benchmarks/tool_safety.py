"""Deterministic end-to-end regression benchmark for tool safety governance."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import tempfile
import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable
from unittest.mock import patch

from benchmarks.providers.scripted import ScriptedProvider
from nanobot.agent.hook import HookManager, LifecycleEventType
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.registry import ToolExecutionResult, ToolRegistry
from nanobot.config.schema import ToolPolicyConfig, ToolPolicyRuleConfig
from nanobot.security.tool_policy import (
    ToolAuditSink,
    ToolPolicyDecision,
    ToolPolicyEngine,
    ToolRequestContext,
)

SCHEMA = "litebot-tool-safety-benchmark/v1"
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "benchmark_results"


@dataclass(slots=True)
class Check:
    name: str
    category: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class CaseOutcome:
    case_name: str
    checks: list[Check]
    metrics: dict[str, Any] = field(default_factory=dict)
    audit_projection: list[dict[str, Any]] = field(default_factory=list)

    @property
    def task_success(self) -> bool:
        return all(check.passed for check in self.checks if check.category == "task")

    @property
    def audit_success(self) -> bool:
        return all(check.passed for check in self.checks if check.category == "audit")

    @property
    def strict_success(self) -> bool:
        return self.task_success and self.audit_success

    @property
    def failure_reason(self) -> str | None:
        failures = [
            f"{check.name}: {check.detail or 'check failed'}"
            for check in self.checks
            if not check.passed
        ]
        return "; ".join(failures) or None

    def as_result(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "task_success": self.task_success,
            "audit_success": self.audit_success,
            "strict_success": self.strict_success,
            "failure_reason": self.failure_reason,
            "checks": [asdict(check) for check in self.checks],
            "metrics": self.metrics,
            "audit_projection": self.audit_projection,
        }


class CountingTool(Tool):
    """Side-effect-free benchmark tool that records exact invocation count."""

    description = "Record a deterministic benchmark invocation."
    parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "left": {"type": "integer"},
            "right": {"type": "integer"},
            "command": {"type": "string"},
        },
    }

    def __init__(
        self,
        name: str = "counting_tool",
        on_execute: Callable[[], None] | None = None,
    ) -> None:
        self._name = name
        self.invocations = 0
        self.arguments: list[dict[str, Any]] = []
        self.on_execute = on_execute

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, **kwargs: Any) -> str:
        self.invocations += 1
        self.arguments.append(dict(kwargs))
        if self.on_execute:
            self.on_execute()
        return f"invocation:{self.invocations}"


class CountingReadFileTool(ReadFileTool):
    """Use the production read_file schema while making execution observable."""

    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace=workspace)
        self.invocations = 0

    async def execute(self, **kwargs: Any) -> str:
        self.invocations += 1
        return "unexpected read execution"


class PartialExecutionTool(Tool):
    name = "partial_execution_tool"
    description = "Record one completed step and report a deterministic partial result."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def __init__(self) -> None:
        self.invocations = 0
        self.effects: list[str] = []

    async def execute(self, **kwargs: Any) -> ToolExecutionResult:
        self.invocations += 1
        self.effects.append("step_1_completed")
        return ToolExecutionResult.partial(
            "step 2 failed",
            completed_steps=["step_1"],
            failed_steps=["step_2"],
            retry_recommended=False,
        )


class FakeApprover:
    def __init__(self, response: bool | BaseException) -> None:
        self.response = response
        self.calls = 0

    async def __call__(
        self, context: ToolRequestContext, decision: ToolPolicyDecision
    ) -> bool:
        self.calls += 1
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class RecordingPolicyEngine(ToolPolicyEngine):
    """Capture immutable decision projections before runner confirmation mutation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.decisions: list[dict[str, Any]] = []

    async def evaluate(self, context: ToolRequestContext) -> ToolPolicyDecision:
        decision = await super().evaluate(context)
        self.decisions.append(
            {
                "tool_call_id": context.tool_call_id,
                "tool": context.name,
                "action": decision.action.value,
                "reason": decision.reason,
                "reason_codes": list(decision.reason_codes),
                "argument_hash": decision.argument_hash,
            }
        )
        return decision


@dataclass(slots=True)
class RunArtifacts:
    invocations: int
    approver_calls: int
    decisions: list[dict[str, Any]]
    audit: list[dict[str, Any]]
    audit_events: list[dict[str, Any]]
    tool_events: list[dict[str, Any]]
    effects: list[str]
    final_content: str | None


def _blocked_external_io(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("tool safety benchmark attempted forbidden external I/O")


def _confirm_config() -> ToolPolicyConfig:
    return ToolPolicyConfig(
        rules=[
            ToolPolicyRuleConfig(
                id="benchmark-confirm",
                tools=["counting_tool"],
                action="confirm",
                reason="benchmark approval required",
                priority=100,
            )
        ]
    )


async def _run_calls(
    case_name: str,
    calls: list[tuple[str, dict[str, Any]]],
    *,
    tool_name: str = "counting_tool",
    config: ToolPolicyConfig | None = None,
    approver: FakeApprover | None = None,
    clock: FakeClock | None = None,
    on_execute: Callable[[], None] | None = None,
    tool: Tool | None = None,
    audit_line_writer: Callable[[str], None] | None = None,
) -> RunArtifacts:
    with tempfile.TemporaryDirectory(prefix=f"tool-safety-{case_name}-") as root:
        workspace = Path(root)
        engine = RecordingPolicyEngine(
            config or ToolPolicyConfig(), workspace, clock=clock or time.monotonic
        )
        if audit_line_writer:
            engine.audit = ToolAuditSink(
                workspace,
                engine.config.audit_path,
                engine.redactor,
                line_writer=audit_line_writer,
            )
        tool = tool or CountingTool(tool_name, on_execute=on_execute)
        tool_name = tool.name
        registry = ToolRegistry()
        registry.register(tool)
        audit_events: list[dict[str, Any]] = []
        hooks = HookManager()
        hooks.register(
            LifecycleEventType.TOOL_AUDIT,
            lambda event: audit_events.append(dict(event.payload)),
            name="tool-safety-audit-recorder",
        )
        provider = ScriptedProvider(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {"id": call_id, "name": tool_name, "arguments": arguments}
                        for call_id, arguments in calls
                    ],
                },
                {"content": "benchmark complete"},
            ]
        )
        with ExitStack() as stack:
            for target in (
                "socket.getaddrinfo",
                "socket.create_connection",
                "httpx.AsyncClient.request",
                "asyncio.create_subprocess_exec",
                "asyncio.create_subprocess_shell",
                "subprocess.Popen",
            ):
                stack.enter_context(patch(target, _blocked_external_io))
            result = await AgentRunner(provider).run(
                AgentRunSpec(
                    initial_messages=[{"role": "user", "content": "fixed safety input"}],
                    tools=registry,
                    model=provider.get_default_model(),
                    max_iterations=3,
                    run_id=f"tool-safety-{case_name}",
                    session_key=f"benchmark:{case_name}",
                    event_metadata={"channel": "benchmark"},
                    policy_engine=engine,
                    confirmation_handler=approver,
                    hook_manager=hooks,
                )
            )
        audit = []
        if engine.audit.path.exists():
            audit = [
                json.loads(line)
                for line in engine.audit.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return RunArtifacts(
            getattr(tool, "invocations", 0),
            approver.calls if approver else 0,
            engine.decisions,
            audit,
            audit_events,
            result.tool_events,
            list(getattr(tool, "effects", [])),
            result.final_content,
        )


def _check(name: str, category: str, condition: bool, detail: Any = "") -> Check:
    passed = bool(condition)
    return Check(name, category, passed, "" if passed else str(detail))


def _audit_projection(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "run_id",
        "session_key",
        "tool_call_id",
        "tool",
        "argument_hash",
        "policy_action",
        "policy_reason",
        "reason_codes",
        "rule_ids",
        "stage",
        "outcome",
        "confirmation",
        "approval_error",
        "progress",
        "error",
    )
    return [{key: record[key] for key in keys if key in record} for record in records]


def _basic_outcome(
    case_name: str,
    artifacts: RunArtifacts,
    task_checks: list[Check],
    audit_checks: list[Check],
) -> CaseOutcome:
    return CaseOutcome(
        case_name,
        task_checks + audit_checks,
        metrics={
            "tool_invocations": artifacts.invocations,
            "approver_calls": artifacts.approver_calls,
            "audit_records": len(artifacts.audit),
            "audit_events": len(artifacts.audit_events),
        },
        audit_projection=_audit_projection(artifacts.audit),
    )


async def safe_allow() -> CaseOutcome:
    data = await _run_calls("safe_allow", [("call-safe", {"value": "safe"})])
    final = data.audit[-1] if data.audit else {}
    return _basic_outcome(
        "safe_allow",
        data,
        [
            _check("policy_allow", "task", [d["action"] for d in data.decisions] == ["allow"], data.decisions),
            _check("executes_once", "task", data.invocations == 1, data.invocations),
            _check("approver_not_called", "task", data.approver_calls == 0, data.approver_calls),
        ],
        [_check("success_audit", "audit", final.get("policy_action") == "allow" and final.get("outcome") == "success", final)],
    )


async def explicit_deny() -> CaseOutcome:
    data = await _run_calls(
        "explicit_deny", [("call-deny", {"command": "rm -rf harmless-fixture"})], tool_name="exec"
    )
    final = data.audit[-1] if data.audit else {}
    return _basic_outcome(
        "explicit_deny",
        data,
        [
            _check("policy_deny", "task", [d["action"] for d in data.decisions] == ["deny"], data.decisions),
            _check("never_executes", "task", data.invocations == 0, data.invocations),
            _check("approver_not_called", "task", data.approver_calls == 0, data.approver_calls),
        ],
        [_check("dangerous_command_audit", "audit", final.get("outcome") == "failed" and "command.dangerous" in final.get("reason_codes", []), final)],
    )


async def confirm_approve() -> CaseOutcome:
    approver = FakeApprover(True)
    data = await _run_calls("confirm_approve", [("call-confirm", {"value": "ok"})], config=_confirm_config(), approver=approver)
    final = data.audit[-1] if data.audit else {}
    return _basic_outcome(
        "confirm_approve",
        data,
        [
            _check("policy_confirm", "task", data.decisions[0]["action"] == "confirm", data.decisions),
            _check("approver_once", "task", data.approver_calls == 1, data.approver_calls),
            _check("executes_once", "task", data.invocations == 1, data.invocations),
        ],
        [_check("approved_success_audit", "audit", len(data.audit) == 2 and final.get("confirmation") == "approved" and final.get("outcome") == "success", data.audit)],
    )


async def confirm_reject() -> CaseOutcome:
    approver = FakeApprover(False)
    data = await _run_calls("confirm_reject", [("call-confirm", {"value": "no"})], config=_confirm_config(), approver=approver)
    final = data.audit[-1] if data.audit else {}
    return _basic_outcome(
        "confirm_reject",
        data,
        [
            _check("policy_confirm", "task", data.decisions[0]["action"] == "confirm", data.decisions),
            _check("approver_once", "task", data.approver_calls == 1, data.approver_calls),
            _check("never_executes", "task", data.invocations == 0, data.invocations),
        ],
        [_check("declined_failed_audit", "audit", len(data.audit) == 2 and final.get("confirmation") == "declined" and final.get("outcome") == "failed", data.audit)],
    )


async def approver_failure_fail_closed() -> CaseOutcome:
    approver = FakeApprover(RuntimeError("approval service unavailable: token=top-secret"))
    data = await _run_calls("approver_failure_fail_closed", [("call-confirm", {"value": "blocked"})], config=_confirm_config(), approver=approver)
    final = data.audit[-1] if data.audit else {}
    serialized = json.dumps(data.audit, ensure_ascii=False)
    return _basic_outcome(
        "approver_failure_fail_closed",
        data,
        [
            _check("approver_once", "task", data.approver_calls == 1, data.approver_calls),
            _check("fails_closed", "task", data.invocations == 0, data.invocations),
        ],
        [
            _check("approval_error_audit", "audit", final.get("confirmation") == "error" and "approval handler errored" in final.get("result_summary", "") and "approval service unavailable" in final.get("approval_error", ""), final),
            _check("approval_error_redacted", "audit", "top-secret" not in serialized and "[REDACTED]" in serialized, serialized),
        ],
    )


async def duplicate_block() -> CaseOutcome:
    data = await _run_calls(
        "duplicate_block",
        [("call-first", {"left": 1, "right": 2}), ("call-second", {"right": 2, "left": 1})],
    )
    final = data.audit[-1] if data.audit else {}
    return _basic_outcome(
        "duplicate_block",
        data,
        [
            _check("allow_then_deny", "task", [d["action"] for d in data.decisions] == ["allow", "deny"], data.decisions),
            _check("executes_once", "task", data.invocations == 1, data.invocations),
        ],
        [_check("duplicate_block_audit", "audit", "duplicate.recent" in final.get("reason_codes", []) and final.get("outcome") == "failed", final)],
    )


async def same_tool_different_args() -> CaseOutcome:
    data = await _run_calls(
        "same_tool_different_args",
        [("call-one", {"value": "one"}), ("call-two", {"value": "two"})],
    )
    reasons = [code for record in data.audit for code in record.get("reason_codes", [])]
    return _basic_outcome(
        "same_tool_different_args",
        data,
        [
            _check("both_allowed", "task", [d["action"] for d in data.decisions] == ["allow", "allow"], data.decisions),
            _check("executes_twice", "task", data.invocations == 2, data.invocations),
        ],
        [_check("no_duplicate_audit", "audit", len(data.audit) == 2 and "duplicate.recent" not in reasons and all(record.get("outcome") == "success" for record in data.audit), data.audit)],
    )


async def duplicate_window_expired() -> CaseOutcome:
    clock = FakeClock()
    data = await _run_calls(
        "duplicate_window_expired",
        [("call-one", {"value": "same"}), ("call-two", {"value": "same"})],
        clock=clock,
        on_execute=lambda: clock.advance(6.0),
    )
    reasons = [code for record in data.audit for code in record.get("reason_codes", [])]
    return _basic_outcome(
        "duplicate_window_expired",
        data,
        [
            _check("both_allowed", "task", [d["action"] for d in data.decisions] == ["allow", "allow"], data.decisions),
            _check("executes_twice", "task", data.invocations == 2, data.invocations),
        ],
        [_check("window_expiry_audit", "audit", len(data.audit) == 2 and "duplicate.recent" not in reasons and all(record.get("outcome") == "success" for record in data.audit), data.audit)],
    )


async def deny_audit_integrity() -> CaseOutcome:
    data = await _run_calls(
        "deny_audit_integrity", [("audit-deny", {"command": "shutdown now"})], tool_name="exec"
    )
    record = data.audit[-1] if data.audit else {}
    try:
        datetime.fromisoformat(record.get("timestamp", ""))
        timestamp_valid = True
    except (TypeError, ValueError):
        timestamp_valid = False
    expected = {
        "run_id": "tool-safety-deny_audit_integrity",
        "session_key": "benchmark:deny_audit_integrity",
        "tool_call_id": "audit-deny",
        "tool": "exec",
        "policy_action": "deny",
        "outcome": "failed",
    }
    return _basic_outcome(
        "deny_audit_integrity",
        data,
        [
            _check("denied_without_execution", "task", data.invocations == 0 and data.decisions[0]["action"] == "deny", data.decisions),
        ],
        [
            _check("correlation_fields", "audit", all(record.get(key) == value for key, value in expected.items()), record),
            _check("reason_and_hash", "audit", bool(record.get("policy_reason")) and "command.dangerous" in record.get("reason_codes", []) and len(record.get("argument_hash", "")) == 64, record),
            _check("timestamp_parseable", "audit", timestamp_valid, record.get("timestamp")),
        ],
    )


async def confirm_audit_chain() -> CaseOutcome:
    approver = FakeApprover(True)
    data = await _run_calls("confirm_audit_chain", [("audit-confirm", {"value": "chain"})], config=_confirm_config(), approver=approver)
    pending = next((record for record in data.audit if record.get("outcome") == "pending"), {})
    final = next((record for record in data.audit if record.get("outcome") == "success"), {})
    correlation = ("run_id", "tool_call_id", "argument_hash")
    return _basic_outcome(
        "confirm_audit_chain",
        data,
        [
            _check("approve_and_execute_once", "task", data.approver_calls == data.invocations == 1, {"approver": data.approver_calls, "tool": data.invocations}),
        ],
        [
            _check("two_stage_chain", "audit", len(data.audit) == 2 and pending.get("stage") == "confirmation" and pending.get("policy_action") == "confirm" and final.get("confirmation") == "approved", data.audit),
            _check("correlation_stable", "audit", bool(pending) and all(pending.get(key) == final.get(key) and pending.get(key) for key in correlation), {"pending": pending, "final": final}),
        ],
    )


def _failing_audit_writer(_line: str) -> None:
    raise OSError("simulated audit write failure")


async def audit_write_failure() -> CaseOutcome:
    denied = await _run_calls(
        "audit_write_failure_deny",
        [("audit-failure-deny", {"command": "shutdown now"})],
        tool_name="exec",
        audit_line_writer=_failing_audit_writer,
    )
    approver = FakeApprover(False)
    rejected = await _run_calls(
        "audit_write_failure_reject",
        [("audit-failure-reject", {"value": "blocked"})],
        config=_confirm_config(),
        approver=approver,
        audit_line_writer=_failing_audit_writer,
    )
    lifecycle = denied.audit_events + rejected.audit_events
    tool_events = denied.tool_events + rejected.tool_events
    errors = [event.get("error", "") for event in lifecycle]
    checks = [
        _check(
            "deny_and_reject_preserved",
            "task",
            [item["action"] for item in denied.decisions] == ["deny"]
            and [item["action"] for item in rejected.decisions] == ["confirm"],
            {"deny": denied.decisions, "reject": rejected.decisions},
        ),
        _check(
            "tools_never_execute",
            "task",
            denied.invocations == rejected.invocations == 0,
            {"deny": denied.invocations, "reject": rejected.invocations},
        ),
        _check("reject_uses_approver_once", "task", rejected.approver_calls == 1, rejected.approver_calls),
        _check(
            "audit_failure_exposed",
            "audit",
            len(lifecycle) == 3
            and all(event.get("degraded") is True for event in lifecycle)
            and all(event.get("write_succeeded") is False for event in lifecycle)
            and all("simulated audit write failure" in error for error in errors),
            lifecycle,
        ),
        _check(
            "tool_events_marked_degraded",
            "audit",
            len(tool_events) == 2
            and all(event.get("audit_status") == "degraded" for event in tool_events)
            and all("simulated audit write failure" in event.get("audit_error", "") for event in tool_events),
            tool_events,
        ),
        _check(
            "failed_writes_not_claimed_as_jsonl",
            "audit",
            denied.audit == rejected.audit == [],
            {"deny": denied.audit, "reject": rejected.audit},
        ),
    ]
    return CaseOutcome(
        "audit_write_failure",
        checks,
        metrics={
            "tool_invocations": denied.invocations + rejected.invocations,
            "approver_calls": rejected.approver_calls,
            "audit_records": 0,
            "degraded_audit_events": len(lifecycle),
        },
        audit_projection=[
            {
                "write_succeeded": event.get("write_succeeded"),
                "degraded": event.get("degraded"),
                "error": event.get("error"),
                "tool_call_id": event.get("record", {}).get("tool_call_id"),
                "policy_action": event.get("record", {}).get("policy_action"),
                "outcome": event.get("record", {}).get("outcome"),
            }
            for event in lifecycle
        ],
    )


async def invalid_tool_arguments() -> CaseOutcome:
    with tempfile.TemporaryDirectory(prefix="tool-safety-invalid-schema-") as root:
        tool = CountingReadFileTool(Path(root))
        data = await _run_calls(
            "invalid_tool_arguments",
            [("invalid-arguments", {})],
            tool=tool,
            config=ToolPolicyConfig(duplicate_window_seconds=0),
        )
    record = data.audit[-1] if data.audit else {}
    error = record.get("error") or {}
    tool_event = data.tool_events[-1] if data.tool_events else {}
    return _basic_outcome(
        "invalid_tool_arguments",
        data,
        [
            _check("policy_allows_validation", "task", data.decisions[0]["action"] == "allow", data.decisions),
            _check("implementation_not_called", "task", data.invocations == 0, data.invocations),
            _check("tool_event_is_error", "task", tool_event.get("status") == "error" and "Invalid parameters" in tool_event.get("detail", ""), tool_event),
        ],
        [
            _check("validation_stage_audit", "audit", record.get("stage") == "validation" and record.get("outcome") == "failed", record),
            _check("invalid_request_taxonomy", "audit", error.get("retry_class") == "invalid_request" and error.get("retryable") is False, error),
            _check("not_security_or_execution_error", "audit", record.get("policy_action") == "allow" and "duplicate.recent" not in record.get("reason_codes", []) and error.get("stage") == "validation", record),
        ],
    )


async def partial_execution_failure() -> CaseOutcome:
    tool = PartialExecutionTool()
    data = await _run_calls(
        "partial_execution_failure",
        [("partial-execution", {"value": "run"})],
        tool=tool,
        config=ToolPolicyConfig(duplicate_window_seconds=0),
    )
    record = data.audit[-1] if data.audit else {}
    tool_event = data.tool_events[-1] if data.tool_events else {}
    progress = {
        "completed_steps": ["step_1"],
        "failed_steps": ["step_2"],
        "retry_recommended": False,
    }
    return _basic_outcome(
        "partial_execution_failure",
        data,
        [
            _check("single_execution_no_retry", "task", data.invocations == 1, data.invocations),
            _check("completed_effect_visible", "task", data.effects == ["step_1_completed"], data.effects),
            _check("tool_event_partial", "task", tool_event.get("status") == "ok" and tool_event.get("outcome") == "partial" and tool_event.get("progress") == progress, tool_event),
        ],
        [
            _check("partial_audit_outcome", "audit", record.get("outcome") == "partial" and record.get("progress") == progress, record),
            _check("not_complete_success", "audit", record.get("outcome") != "success" and record.get("error") is None, record),
        ],
    )


CASES: dict[str, Callable[[], Awaitable[CaseOutcome]]] = {
    "safe_allow": safe_allow,
    "explicit_deny": explicit_deny,
    "confirm_approve": confirm_approve,
    "confirm_reject": confirm_reject,
    "approver_failure_fail_closed": approver_failure_fail_closed,
    "duplicate_block": duplicate_block,
    "same_tool_different_args": same_tool_different_args,
    "duplicate_window_expired": duplicate_window_expired,
    "deny_audit_integrity": deny_audit_integrity,
    "confirm_audit_chain": confirm_audit_chain,
    "audit_write_failure": audit_write_failure,
    "invalid_tool_arguments": invalid_tool_arguments,
    "partial_execution_failure": partial_execution_failure,
}


async def run_benchmark(*, case: str | None = None) -> dict[str, Any]:
    selected = [case] if case else list(CASES)
    if any(name not in CASES for name in selected):
        raise ValueError(f"unknown tool safety benchmark case: {case}")
    outcomes: list[CaseOutcome] = []
    for name in selected:
        started = time.perf_counter()
        try:
            outcome = await CASES[name]()
        except BaseException as exc:
            outcome = CaseOutcome(
                name,
                [
                    Check("case_execution", "task", False, f"{type(exc).__name__}: {exc}"),
                    Check("audit_unavailable", "audit", False, "case did not complete"),
                ],
            )
        outcome.metrics["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        outcomes.append(outcome)
    results = [outcome.as_result() for outcome in outcomes]
    count = len(outcomes)
    return {
        "schema_version": SCHEMA,
        "created_at": datetime.now().isoformat(),
        "python": platform.python_version(),
        "cases": count,
        "passed": sum(outcome.strict_success for outcome in outcomes),
        "failed": sum(not outcome.strict_success for outcome in outcomes),
        "task_success_rate": sum(outcome.task_success for outcome in outcomes) / count,
        "audit_success_rate": sum(outcome.audit_success for outcome in outcomes) / count,
        "strict_success_rate": sum(outcome.strict_success for outcome in outcomes) / count,
        "results": results,
    }


def _print_summary(report: dict[str, Any]) -> None:
    for result in report["results"]:
        status = "PASSED" if result["strict_success"] else "FAILED"
        print(f"{status:7} {result['case_name']}")
        if result["failure_reason"]:
            print(f"        {result['failure_reason']}")
    print("\nTool Safety Regression Summary\n")
    print(f"Cases:               {report['cases']}")
    print(f"Passed:              {report['passed']}")
    print(f"Failed:              {report['failed']}")
    print(f"Task Success Rate:   {report['task_success_rate'] * 100:.1f}%")
    print(f"Audit Success Rate:  {report['audit_success_rate'] * 100:.1f}%")
    print(f"Strict Success Rate: {report['strict_success_rate'] * 100:.1f}%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiteBot deterministic tool safety benchmark")
    parser.add_argument("--case", choices=tuple(CASES))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = asyncio.run(run_benchmark(case=args.case))
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-tool-safety-{uuid.uuid4().hex[:8]}"
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
    return 0 if report["strict_success_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
