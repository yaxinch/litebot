"""Strict live-provider A/B benchmark for long-context management."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import statistics
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from benchmarks.collector import BenchmarkCollector, RecordingProvider, RecordingTool, add_usage
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.tool_result import (
    ArtifactRetrievalGuard,
    GetToolResultTool,
    SearchToolResultTool,
)
from nanobot.bus.queue import MessageBus
from nanobot.cli.commands import _make_provider
from nanobot.config.loader import load_config
from nanobot.config.schema import ContextManagementConfig
from nanobot.providers.base import GenerationSettings
from nanobot.session.manager import SessionManager

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "benchmark_results"
Mode = Literal["baseline", "context_management"]
SUITES = ("long-context", "large-tool-result")
EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
RETRIEVAL_TOOLS = {"get_tool_result", "search_tool_result"}
ARTIFACT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _filler(label: str, chars: int) -> str:
    sentence = f"{label}: discuss interfaces, tests, compatibility, rollout, and verification without changing recorded facts. "
    return (sentence * math.ceil(chars / len(sentence)))[:chars]


@dataclass(frozen=True, slots=True)
class ABCase:
    id: str
    suite: str
    description: str
    seed_messages: tuple[tuple[str, str], ...]
    turns: tuple[str, ...]
    required: tuple[str, ...]
    forbidden: tuple[str, ...] = ()
    early_markers: tuple[str, ...] = ()
    fact_markers: tuple[str, ...] = ()
    requires_retrieval: bool = False
    fixture_kind: str | None = None


def _seed(marker: str, turns: int, chars: int) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = [
        ("user", f"Record this exact early fact and never alter it: {marker}."),
        ("assistant", f"Recorded {marker}."),
    ]
    for index in range(turns):
        rows.extend((("user", _filler(f"historical-{index}", chars)), ("assistant", f"Historical step {index} complete; constraints unchanged.")))
    return tuple(rows)


def cases_for(suite: str) -> list[ABCase]:
    common_final = "Reply concisely and include every requested exact marker. Do not invent any marker."
    if suite == "large-tool-result":
        return [
            ABCase(
                "large_cross_turn", suite, "Large deterministic tool result retained across a later turn.", (),
                (f"What exact marker was in the earlier large report? {common_final}",),
                ("LARGE-RESULT-993",), fact_markers=("LARGE-RESULT-993",), fixture_kind="large",
            ),
            ABCase(
                "large_paged_retrieval", suite, "Retrieve a marker beyond the artifact preview.", (),
                ("Find and return the exact marker from the earlier report. It may require reading additional stored result pages.",),
                ("PAGED-RESULT-771",), fact_markers=("PAGED-RESULT-771",), requires_retrieval=True, fixture_kind="paged",
            ),
        ]
    return [
        ABCase(
            "ordinary_multi_turn", suite, "Ordinary conversation grows over real model turns.", (),
            tuple([f"Remember ORDINARY-ALPHA-101. {_filler(f'ordinary-{i}', 6500)} Reply ACK-{i}." for i in range(4)] +
                  [f"Return ORDINARY-ALPHA-101 and ACK-3. {common_final}"]),
            ("ORDINARY-ALPHA-101", "ACK-3"), early_markers=("ORDINARY-ALPHA-101",), fact_markers=("ACK-3",),
        ),
        ABCase(
            "multi_turn_tools_medium", suite, "Multiple agent turns with a medium fixed tool result.", _seed("MEDIUM-CONSTRAINT-202", 5, 4500),
            (f"Return MEDIUM-CONSTRAINT-202 and the exact marker from the earlier medium report. {common_final}",),
            ("MEDIUM-CONSTRAINT-202", "MEDIUM-RESULT-303"), early_markers=("MEDIUM-CONSTRAINT-202",), fact_markers=("MEDIUM-RESULT-303",), fixture_kind="medium",
        ),
        ABCase(
            "single_summary", suite, "Cross the threshold once and recall the first constraint.", _seed("ONE-SUMMARY-404", 10, 5600),
            (f"Return ONE-SUMMARY-404. {common_final}",), ("ONE-SUMMARY-404",), early_markers=("ONE-SUMMARY-404",),
        ),
        ABCase(
            "multiple_summaries", suite, "Add enough later work to compact more than once.", _seed("MULTI-SUMMARY-505", 10, 4200),
            (f"{_filler('new-wave-a', 26000)} Reply WAVE-A.", f"{_filler('new-wave-b', 26000)} Reply WAVE-B.",
             f"Return MULTI-SUMMARY-505, WAVE-A, and WAVE-B. {common_final}"),
            ("MULTI-SUMMARY-505", "WAVE-A", "WAVE-B"), early_markers=("MULTI-SUMMARY-505",), fact_markers=("WAVE-A", "WAVE-B"),
        ),
        ABCase(
            "early_constraint_recall", suite, "Recall exact early constraint after compaction.", _seed("CONSTRAINT-NO-SQLITE-606", 10, 5600),
            (f"State the exact implementation constraint CONSTRAINT-NO-SQLITE-606. {common_final}",),
            ("CONSTRAINT-NO-SQLITE-606",), forbidden=("CONSTRAINT-USE-SQLITE",), early_markers=("CONSTRAINT-NO-SQLITE-606",),
        ),
        ABCase(
            "post_compaction_multistep", suite, "Continue an ordered task after old context is compacted.",
            _seed("PLAN-ORDER-707", 10, 5600) + (("user", "Required order is DISCOVER-1 then BUILD-2 then VERIFY-3."), ("assistant", "Order recorded.")),
            (f"Complete the plan by returning PLAN-ORDER-707 then DISCOVER-1 then BUILD-2 then VERIFY-3 in that order. {common_final}",),
            ("PLAN-ORDER-707", "DISCOVER-1", "BUILD-2", "VERIFY-3"), early_markers=("PLAN-ORDER-707",), fact_markers=("DISCOVER-1", "BUILD-2", "VERIFY-3"),
        ),
        ABCase(
            "recent_turn_preservation", suite, "Preserve summarized early and uncompressed recent facts.",
            _seed("EARLY-FACT-808", 10, 5600) + (("user", "The newest exact instruction is RECENT-INSTRUCTION-909."), ("assistant", "Recorded RECENT-INSTRUCTION-909.")),
            (f"Return EARLY-FACT-808 and RECENT-INSTRUCTION-909. {common_final}",),
            ("EARLY-FACT-808", "RECENT-INSTRUCTION-909"), early_markers=("EARLY-FACT-808",), fact_markers=("RECENT-INSTRUCTION-909",),
        ),
    ]


def fixture_payload(kind: str) -> str:
        if kind == "medium":
            return _filler("medium-report", 6000) + "\nMEDIUM-RESULT-303"
        if kind == "large":
            return _filler("large-report", 12000) + "\nLARGE-RESULT-993\n" + _filler("large-tail", 48000)
        return _filler("paged-prefix", 12000) + "\nPAGED-RESULT-771\n" + _filler("paged-tail", 50000)


def _retrieval_events(tool_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in tool_events if event.get("name") in RETRIEVAL_TOOLS]


def _is_sequential_scan(events: list[dict[str, Any]]) -> bool:
    pages = [
        event for event in events
        if event.get("name") == "get_tool_result" and event.get("status") == "ok"
    ]
    consecutive = 1
    previous: tuple[str, int, int] | None = None
    for event in pages:
        arguments = event.get("arguments", {})
        try:
            current = (
                str(event.get("artifact_id") or arguments.get("artifact_id", "")),
                int(arguments.get("offset", 0)),
                int(event.get("returned_artifact_chars", 0)),
            )
        except (TypeError, ValueError):
            previous = None
            consecutive = 1
            continue
        if previous and current[0] == previous[0] and current[1] == previous[1] + previous[2]:
            consecutive += 1
            if consecutive >= 3:
                return True
        else:
            consecutive = 1
        previous = current
    return any(
        event.get("guard_decision") == "blocked"
        and "sequential" in str(event.get("detail", "")).lower()
        for event in events
    )


def _retrieval_path(events: list[dict[str, Any]]) -> str:
    if not events:
        return "none"
    if _is_sequential_scan(events):
        return "sequential_scan"
    names = [event.get("name") for event in events]
    has_search = "search_tool_result" in names
    has_get = "get_tool_result" in names
    if has_search and not has_get:
        return "search_only"
    if has_get and not has_search:
        return "get_only"
    first_search = names.index("search_tool_result")
    if any(name == "get_tool_result" for name in names[first_search + 1:]):
        return "search_then_get"
    return "get_then_search"


def _retrieval_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    events = _retrieval_events(row.get("tool_events", []))
    artifact_chars = _artifact_chars_for_row(row)
    token_sources = {
        str(event.get("retrieval_token_cost_source", "byte_heuristic"))
        for event in events
    }
    token_cost = sum(
        int(event.get("retrieval_token_cost_estimated") or (
            (int(event.get("result_size_bytes", 0)) + 3) // 4
        ))
        for event in events
    )
    carried_tokens = sum(
        int(item.get("request", {}).get("artifact_retrieval_tokens_estimated") or (
            (int(item.get("request", {}).get("artifact_retrieval_chars", 0)) + 3) // 4
        ))
        for item in row.get("rounds", [])
    )
    retrieval_errors = sum(
        event.get("status") in {"error_result", "exception"} for event in events
    )
    guard_triggers = sum(event.get("guard_decision") == "blocked" for event in events)
    safety_violations = _retrieval_safety_violations(events)
    return {
        "retrieval_path": _retrieval_path(events),
        "get_tool_result_calls": sum(event.get("name") == "get_tool_result" for event in events),
        "search_tool_result_calls": sum(event.get("name") == "search_tool_result" for event in events),
        "artifact_retrieval_count": len(events),
        "artifact_chars_returned_to_model": artifact_chars,
        "retrieval_token_cost_estimated": token_cost,
        "retrieval_token_cost_source": (
            next(iter(token_sources)) if len(token_sources) == 1 else "mixed"
        ) if token_sources else "none",
        "retrieval_prompt_tokens_carried_estimated": carried_tokens,
        "retrieval_guard_trigger_count": guard_triggers,
        "retrieval_error_count": retrieval_errors,
        "safety_violations": dict(sorted(safety_violations.items())),
        "invalid_artifact_call_count": sum(
            safety_violations[kind]
            for kind in ("invalid_artifact_id", "artifact_unavailable")
        ),
    }


def _retrieval_safety_violations(events: list[dict[str, Any]]) -> Counter[str]:
    """Classify explicit retrieval violations without conflating them with task quality."""
    violations: Counter[str] = Counter()
    for event in events:
        detail = str(event.get("detail", "")).lower()
        if event.get("guard_decision") == "blocked":
            violations["retrieval_guard_violation"] += 1
        elif "invalid artifact_id" in detail or "invalid artifact id" in detail:
            violations["invalid_artifact_id"] += 1
        elif "artifact unavailable" in detail or "artifact not found" in detail:
            violations["artifact_unavailable"] += 1
        elif "does not belong to this session" in detail or "isolation" in detail:
            violations["artifact_isolation_violation"] += 1
        elif "path" in detail and event.get("status") in {"error_result", "exception"}:
            violations["path_safety_violation"] += 1
        elif event.get("status") == "exception":
            violations["retrieval_exception"] += 1
        elif event.get("status") == "error_result":
            violations["other_retrieval_error"] += 1
    return violations


def _quality(
    case: ABCase, mode: Mode, output: str, tool_events: list[dict[str, Any]],
    overflow_count: int = 0, finish_reason: str | None = None,
) -> dict[str, Any]:
    positions = [output.find(marker) for marker in case.required]
    successful_gets = [
        event for event in tool_events
        if event["name"] == "get_tool_result" and event.get("status") == "ok"
    ]
    protocol_match = (
        not case.requires_retrieval
        or mode == "baseline"
        or bool(successful_gets)
    )
    ordered = positions == sorted(positions) if case.id == "post_compaction_multistep" else True
    marker_correct = all(pos >= 0 for pos in positions) and ordered
    unexpected_markers: set[str] = set()
    if case.suite == "large-tool-result":
        observed = set(re.findall(r"\b(?:LARGE|PAGED)-RESULT-\d+\b", output))
        unexpected_markers = observed.difference(case.required)
    hallucination = bool(unexpected_markers) or any(marker in output for marker in case.forbidden)
    lowered = output.strip().lower()
    output_error = (
        not lowered
        or finish_reason == "error"
        or lowered.startswith("error")
        or "maximum number of tool call iterations" in lowered
    )
    retrieval_events = _retrieval_events(tool_events)
    safety_violations = _retrieval_safety_violations(retrieval_events)
    safety_compliance = not safety_violations
    result = {
        "marker_correct": marker_correct,
        "output_error_detected": output_error,
        "early_constraint_retention": all(marker in output for marker in case.early_markers),
        "key_fact_retention": all(marker in output for marker in case.fact_markers),
        "safety_compliance": safety_compliance,
        "retrieval_safety_ok": safety_compliance,
        "safety_violations": dict(sorted(safety_violations.items())),
        "invalid_artifact_call_count": sum(
            safety_violations[kind]
            for kind in ("invalid_artifact_id", "artifact_unavailable")
        ),
        "retrieval_protocol_match": protocol_match,
        "tool_result_retrieval_correctness": protocol_match,
        "hallucination_detected": hallucination,
    }
    result["task_success"] = (
        marker_correct and not output_error and not hallucination
        and overflow_count == 0
    )
    result["strict_success"] = result["task_success"] and safety_compliance
    return result


def _rescore_row(row: dict[str, Any], case: ABCase) -> dict[str, Any]:
    if "output" not in row:
        row.update(_retrieval_diagnostics(row))
        quality = row.setdefault("quality", {})
        safety = bool(quality.get(
            "safety_compliance",
            quality.get("retrieval_safety_ok", not row["safety_violations"]),
        ))
        quality["safety_compliance"] = safety
        quality["retrieval_safety_ok"] = safety
        quality["strict_success"] = bool(quality.get("task_success")) and safety
        quality.setdefault("safety_violations", row["safety_violations"])
        quality.setdefault("invalid_artifact_call_count", row["invalid_artifact_call_count"])
        return row
    finish_reason = next((
        item.get("finish_reason") for item in reversed(row.get("rounds", []))
        if item.get("phase", "agent") == "agent"
    ), None)
    row.update(_retrieval_diagnostics(row))
    row["quality"] = _quality(
        case, row["mode"], str(row.get("output", "")), row.get("tool_events", []),
        int(row.get("context_overflow_count", 0)), finish_reason,
    )
    row["schema_version"] = "litebot-context-ab-result/v3"
    row["quality_schema_version"] = "litebot-context-quality/v3"
    return row


def _rescore_rows(rows: list[dict[str, Any]], suite: str) -> list[dict[str, Any]]:
    by_id = {case.id: case for case in cases_for(suite)}
    return [
        _rescore_row(row, by_id[row["case_id"]]) if row.get("case_id") in by_id else row
        for row in rows
    ]


class _BaselineUnavailableRetrievalTool(Tool):
    """Keep the candidate schema while preventing baseline artifact access."""

    def __init__(self, inner: Tool):
        self.inner = inner

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def description(self) -> str:
        return self.inner.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self.inner.parameters

    def set_context(self, session_key: str) -> None:
        # Baseline deliberately has no Context Management artifact namespace.
        del session_key

    async def execute(self, **kwargs: Any) -> str:
        artifact_id = str(kwargs.get("artifact_id", ""))
        if not ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
            return "Error: invalid artifact id"
        return "Error: artifact unavailable in baseline mode"


def _seed_session(manager: SessionManager, loop: AgentLoop, collector: BenchmarkCollector, case: ABCase, key: str) -> None:
    session = manager.get_or_create(key)
    for role, content in case.seed_messages:
        session.add_message(role, content)
    if case.fixture_kind:
        kind = case.fixture_kind
        payload = fixture_payload(kind)
        call_id = f"seed-{kind}"
        user = {"role": "user", "content": f"Inspect the deterministic {kind} report."}
        assistant = {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "fixture_report", "arguments": json.dumps({"kind": kind})}}],
        }
        tool = {"role": "tool", "tool_call_id": call_id, "name": "fixture_report", "content": payload}
        loop._save_turn(session, [user, assistant, tool], start_message=user)
        arguments_json = json.dumps({"kind": kind}, sort_keys=True, ensure_ascii=False)
        collector.tool_events.append({
            "name": "fixture_report", "status": "ok", "arguments": {"kind": kind},
            "arguments_sha256": hashlib.sha256(arguments_json.encode()).hexdigest(),
            "result_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "result_size_bytes": len(payload.encode()), "detail": "seeded deterministic fixture", "latency_ms": 0,
        })
    manager.save(session)


async def run_once(case: ABCase, mode: Mode, repetition: int, order: int, settings: dict[str, Any], provider: Any) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    collector = BenchmarkCollector()
    with tempfile.TemporaryDirectory(prefix=f"nanobot-ab-{case.id}-{mode}-") as root:
        workspace = Path(root)
        sessions = SessionManager(workspace)
        key = f"benchmark:{case.id}"
        provider.generation = GenerationSettings(temperature=0, max_tokens=8192, reasoning_effort=None)
        recorded = RecordingProvider(provider, collector)
        loop = AgentLoop(
            bus=MessageBus(), provider=recorded, workspace=workspace,
            model=settings["model"], max_iterations=settings["max_iterations"],
            context_window_tokens=16_384, restrict_to_workspace=True,
            session_manager=sessions, benchmark_context_mode=mode,
            context_management_config=ContextManagementConfig(
                output_reserve_tokens=2048,
                tool_summary_max_chars=settings["tool_summary_max_chars"],
                artifact_search_total_snippet_chars=settings["artifact_search_total_snippet_chars"],
            ),
        )
        registry = ToolRegistry()
        retrieval_guard = ArtifactRetrievalGuard(
            max_reads=loop.context_policy.artifact_max_reads_per_session,
            max_searches=loop.context_policy.artifact_max_searches_per_session,
            max_returned_chars=loop.context_policy.artifact_max_returned_chars_per_session,
            max_sequential_reads=loop.context_policy.artifact_max_sequential_reads,
        )
        get_tool: Tool = GetToolResultTool(
            loop.artifacts, loop.context_policy.artifact_page_size, retrieval_guard,
        )
        search_tool: Tool = SearchToolResultTool(
            loop.artifacts, retrieval_guard,
            total_snippet_chars=loop.context_policy.artifact_search_total_snippet_chars,
        )
        if mode == "baseline":
            get_tool = _BaselineUnavailableRetrievalTool(get_tool)
            search_tool = _BaselineUnavailableRetrievalTool(search_tool)
        registry.register(RecordingTool(get_tool, collector))
        registry.register(RecordingTool(search_tool, collector))
        loop.tools = registry
        _seed_session(sessions, loop, collector, case, key)
        outputs: list[str] = []
        try:
            for turn in case.turns:
                response = await loop.process_direct(turn, session_key=key, channel="benchmark", chat_id=case.id)
                outputs.append(response.content if response else "")
        finally:
            await loop.close_mcp()
            loop.stop()
        output = outputs[-1] if outputs else ""
        usage = collector.usage
        summary_usage = dict(EMPTY_USAGE)
        for item in collector.rounds:
            if item.get("phase") == "rolling_summary":
                summary_usage = add_usage(summary_usage, item["usage"])
        context_events = list(loop.context_manager.telemetry)
        control_tools = [event for event in collector.tool_events if event["name"] not in {"get_tool_result", "search_tool_result"}]
        overflow_count = sum("context_overflow" in event.get("actions", []) for event in context_events)
        overflow_count += sum(item.get("finish_reason") == "error" and "context" in str(outputs).lower() for item in collector.rounds)
        request_signatures = sorted({
            (item["request"]["model"], item["request"]["temperature"], item["request"]["max_tokens"], item["request"]["tool_schema_sha256"])
            for item in collector.rounds if item.get("phase") == "agent"
        })
        row = {
            "schema_version": "litebot-context-ab-result/v1", "case_id": case.id, "suite": case.suite,
            "mode": mode, "repetition": repetition, "order": order, "started_at": started_at,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "provider": settings["provider"], "model": settings["model"], "temperature": 0,
            "max_tokens": 8192, "context_window_tokens": 16_384, "max_iterations": settings["max_iterations"],
            "tool_summary_max_chars": settings["tool_summary_max_chars"],
            "artifact_search_total_snippet_chars": settings["artifact_search_total_snippet_chars"],
            "usage": usage, "usage_complete": collector.usage_complete,
            "peak_prompt_tokens": collector.peak_prompt_tokens, "model_call_count": len(collector.rounds),
            "compaction_count": sum(bool(event.get("compacted_turns")) for event in context_events),
            "compacted_turns": sum(int(event.get("compacted_turns", 0)) for event in context_events),
            "hard_truncated_turns": sum(int(event.get("hard_truncated_turns", 0)) for event in context_events),
            "rolling_summary_usage": summary_usage, "context_overflow_count": overflow_count,
            "offloaded_artifacts": loop.context_manager.total_offloaded_artifacts,
            "artifact_retrieval_count": sum(event["name"] in {"get_tool_result", "search_tool_result"} for event in collector.tool_events),
            "get_tool_result_calls": sum(event["name"] == "get_tool_result" for event in collector.tool_events),
            "search_tool_result_calls": sum(event["name"] == "search_tool_result" for event in collector.tool_events),
            "artifact_chars_returned_to_model": sum(
                int(event.get("returned_artifact_chars", 0)) for event in collector.tool_events
            ),
            "artifact_context_chars_across_model_calls": sum(
                int(item.get("request", {}).get("artifact_retrieval_chars", 0))
                for item in collector.rounds
            ),
            "output": output,
            "request_signatures": [list(item) for item in request_signatures],
            "control_tool_events": control_tools, "tool_events": collector.tool_events,
            "context_events": context_events, "rounds": collector.rounds,
        }
        return _rescore_row(row, case)


def _tool_signature(row: dict[str, Any]) -> list[tuple[str, str | None, str | None]]:
    return [(event["name"], event.get("arguments_sha256"), event.get("result_sha256")) for event in row["control_tool_events"]]


def validate_pair(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for key in (
        "case_id", "repetition", "provider", "model", "temperature", "max_tokens",
        "context_window_tokens", "max_iterations", "tool_summary_max_chars",
        "artifact_search_total_snippet_chars",
    ):
        if baseline.get(key) != candidate.get(key):
            violations.append(f"{key}_mismatch")
    base_agent = [tuple(item) for item in baseline["request_signatures"]]
    new_agent = [tuple(item) for item in candidate["request_signatures"]]
    base_configs = {(item[0], item[1], item[2], item[3]) for item in base_agent}
    new_configs = {(item[0], item[1], item[2], item[3]) for item in new_agent}
    if base_configs != new_configs:
        violations.append("agent_request_configuration_mismatch")
    if _tool_signature(baseline) != _tool_signature(candidate):
        violations.append("fixed_tool_io_mismatch")
    if not baseline["usage_complete"] or not candidate["usage_complete"]:
        violations.append("incomplete_provider_usage")
    return violations


def aggregate(rows: list[dict[str, Any]], suite: str, repetitions: int) -> dict[str, Any]:
    rows = _rescore_rows(rows, suite)
    cases = cases_for(suite)
    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault((row["case_id"], row["repetition"]), {})[row["mode"]] = row
    valid_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for case in cases:
        for repetition in range(repetitions):
            pair = by_pair.get((case.id, repetition), {})
            if set(pair) != {"baseline", "context_management"}:
                issues.append({"case_id": case.id, "repetition": repetition, "violations": ["incomplete_pair"]})
                continue
            violations = validate_pair(pair["baseline"], pair["context_management"])
            if violations:
                issues.append({"case_id": case.id, "repetition": repetition, "violations": violations})
            else:
                valid_rows.extend(pair.values())

    case_rows: list[dict[str, Any]] = []
    for case in cases:
        modes = {mode: [row for row in valid_rows if row["case_id"] == case.id and row["mode"] == mode] for mode in ("baseline", "context_management")}
        totals = {mode: sum(row["usage"]["total_tokens"] for row in items) for mode, items in modes.items()}
        reduction = None if not totals["baseline"] else (totals["baseline"] - totals["context_management"]) / totals["baseline"] * 100

        def distribution(items: list[dict[str, Any]]) -> dict[str, Any]:
            total_values = [int(row["usage"]["total_tokens"]) for row in items]
            prompt_values = [int(row["usage"]["prompt_tokens"]) for row in items]

            def values_stats(values: list[int]) -> dict[str, float | int | None]:
                return {
                    "mean": statistics.mean(values) if values else None,
                    "median": statistics.median(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }

            return {
                "total_tokens": values_stats(total_values),
                "prompt_tokens": values_stats(prompt_values),
                "model_calls": sum(int(row.get("model_call_count", 0)) for row in items),
                "search_calls": sum(int(row.get("search_tool_result_calls", 0)) for row in items),
                "get_calls": sum(int(row.get("get_tool_result_calls", 0)) for row in items),
                "artifact_chars": sum(int(row.get("artifact_chars_returned_to_model", 0)) for row in items),
                "overflow": sum(int(row.get("context_overflow_count", 0)) for row in items),
                "retrieval_paths": dict(sorted(Counter(
                    row.get("retrieval_path", "none") for row in items
                ).items())),
                "preview_chars": sorted({row.get("tool_summary_max_chars") for row in items}),
            }
        case_rows.append({
            "case_id": case.id, "valid_repetitions": len(modes["baseline"]),
            "baseline_total_tokens": totals["baseline"], "context_management_total_tokens": totals["context_management"],
            "reduction_percent": reduction,
            "baseline_success": sum(row["quality"]["task_success"] for row in modes["baseline"]),
            "context_management_success": sum(row["quality"]["task_success"] for row in modes["context_management"]),
            "baseline_safety_compliance": sum(row["quality"]["safety_compliance"] for row in modes["baseline"]),
            "context_management_safety_compliance": sum(row["quality"]["safety_compliance"] for row in modes["context_management"]),
            "baseline_strict_success": sum(row["quality"]["strict_success"] for row in modes["baseline"]),
            "context_management_strict_success": sum(row["quality"]["strict_success"] for row in modes["context_management"]),
            "baseline_distribution": distribution(modes["baseline"]),
            "context_management_distribution": distribution(modes["context_management"]),
        })

    def mode_stats(mode: str) -> dict[str, Any]:
        selected = [row for row in valid_rows if row["mode"] == mode]
        usage = dict(EMPTY_USAGE)
        summary_usage = dict(EMPTY_USAGE)
        for row in selected:
            usage = add_usage(usage, row["usage"])
            summary_usage = add_usage(summary_usage, row["rolling_summary_usage"])
        quality_keys = (
            "task_success", "safety_compliance", "strict_success",
            "marker_correct", "early_constraint_retention",
            "key_fact_retention", "retrieval_safety_ok", "retrieval_protocol_match",
            "tool_result_retrieval_correctness",
        )

        def quality_value(row: dict[str, Any], key: str) -> bool:
            quality = row["quality"]
            if key in quality:
                return bool(quality[key])
            if key == "retrieval_protocol_match":
                return bool(quality.get("tool_result_retrieval_correctness"))
            return bool(quality.get("task_success"))

        path_distribution = Counter(
            row.get("retrieval_path", "none") for row in selected
        )
        token_sources = Counter(
            row.get("retrieval_token_cost_source", "none") for row in selected
        )
        safety_violations: Counter[str] = Counter()
        for row in selected:
            safety_violations.update(row.get("safety_violations", {}))
        return {
            "runs": len(selected), "usage": usage, "peak_prompt_tokens": max((row["peak_prompt_tokens"] for row in selected), default=0),
            "model_call_count": sum(row["model_call_count"] for row in selected),
            "compaction_count": sum(row["compaction_count"] for row in selected), "rolling_summary_usage": summary_usage,
            "context_overflow_count": sum(row["context_overflow_count"] for row in selected),
            "offloaded_artifacts": sum(row["offloaded_artifacts"] for row in selected),
            "artifact_retrieval_count": sum(row["artifact_retrieval_count"] for row in selected),
            "get_tool_result_calls": sum(row.get("get_tool_result_calls", row["artifact_retrieval_count"]) for row in selected),
            "search_tool_result_calls": sum(row.get("search_tool_result_calls", 0) for row in selected),
            "artifact_chars_returned_to_model": sum(row.get("artifact_chars_returned_to_model", 0) for row in selected),
            "artifact_context_chars_across_model_calls": sum(row.get("artifact_context_chars_across_model_calls", 0) for row in selected),
            "retrieval_path_distribution": dict(sorted(path_distribution.items())),
            "retrieval_token_cost_estimated": sum(row.get("retrieval_token_cost_estimated", 0) for row in selected),
            "retrieval_token_cost_sources": dict(sorted(token_sources.items())),
            "retrieval_prompt_tokens_carried_estimated": sum(
                row.get("retrieval_prompt_tokens_carried_estimated", 0) for row in selected
            ),
            "retrieval_guard_trigger_count": sum(row.get("retrieval_guard_trigger_count", 0) for row in selected),
            "retrieval_error_count": sum(row.get("retrieval_error_count", 0) for row in selected),
            "safety_violations": dict(sorted(safety_violations.items())),
            "invalid_artifact_call_count": sum(
                int(row.get("invalid_artifact_call_count", 0)) for row in selected
            ),
            "quality_rates": {
                key: (sum(quality_value(row, key) for row in selected) / len(selected) * 100 if selected else None)
                for key in quality_keys
            },
            "hallucination_rate": (sum(row["quality"]["hallucination_detected"] for row in selected) / len(selected) * 100 if selected else None),
        }

    stats = {mode: mode_stats(mode) for mode in ("baseline", "context_management")}
    reductions = [row["reduction_percent"] for row in case_rows if row["reduction_percent"] is not None and row["valid_repetitions"] == repetitions]
    complete = not issues and len(reductions) == len(cases)
    baseline_total = stats["baseline"]["usage"]["total_tokens"]
    candidate_total = stats["context_management"]["usage"]["total_tokens"]
    baseline_prompt = stats["baseline"]["usage"]["prompt_tokens"]
    candidate_prompt = stats["context_management"]["usage"]["prompt_tokens"]
    metrics = {
        "prompt_token_reduction_percent": ((baseline_prompt - candidate_prompt) / baseline_prompt * 100) if complete and baseline_prompt else None,
        "overall_total_token_reduction_percent": ((baseline_total - candidate_total) / baseline_total * 100) if complete and baseline_total else None,
        "mean_per_case_reduction_percent": statistics.mean(reductions) if complete else None,
        "median_per_case_reduction_percent": statistics.median(reductions) if complete else None,
    }
    return {
        "schema_version": "litebot-context-ab-summary/v3", "suite": suite, "repetitions": repetitions,
        "conclusion_available": complete, "issues": issues, "cases": case_rows, "modes": stats, "metrics": metrics,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [f"# Context Management A/B Benchmark: {summary['suite']}", "", "|Case|Baseline Total Tokens|New Total Tokens|Reduction|Baseline Success|New Success|", "|---|--:|--:|--:|--:|--:|"]
    for row in summary["cases"]:
        reduction = "N/A" if row["reduction_percent"] is None else f"{row['reduction_percent']:.2f}%"
        denominator = row["valid_repetitions"]
        lines.append(f"|{row['case_id']}|{row['baseline_total_tokens']}|{row['context_management_total_tokens']}|{reduction}|{row['baseline_success']}/{denominator}|{row['context_management_success']}/{denominator}|")
    if "tool_summary_max_chars" in summary:
        lines += [
            "",
            f"Preview configuration: {summary['tool_summary_max_chars']} chars; "
            f"search snippet cap: {summary['artifact_search_total_snippet_chars']} chars.",
            "",
            "|Case / Mode|Total mean|Median|Min–max|Prompt mean|Model calls|Search / Get|Task / Safety / Strict|Overflow|Retrieval paths|",
            "|---|--:|--:|--:|--:|--:|--:|--:|--:|---|",
        ]
        for row in summary["cases"]:
            for mode, label, prefix in (
                ("baseline", "Baseline", "baseline"),
                ("context_management", "Context Management", "context_management"),
            ):
                data = row[f"{prefix}_distribution"]
                total = data["total_tokens"]
                prompt = data["prompt_tokens"]
                denominator = row["valid_repetitions"]
                task = row[f"{prefix}_success"]
                safety = row[f"{prefix}_safety_compliance"]
                strict = row[f"{prefix}_strict_success"]
                total_mean = "N/A" if total["mean"] is None else f"{total['mean']:.1f}"
                total_median = "N/A" if total["median"] is None else f"{total['median']:.1f}"
                total_range = (
                    "N/A" if total["min"] is None
                    else f"{total['min']}–{total['max']}"
                )
                prompt_mean = "N/A" if prompt["mean"] is None else f"{prompt['mean']:.1f}"
                lines.append(
                    f"|{row['case_id']} / {label}|{total_mean}|{total_median}|"
                    f"{total_range}|{prompt_mean}|{data['model_calls']}|"
                    f"{data['search_calls']} / {data['get_calls']}|"
                    f"{task}/{denominator} / {safety}/{denominator} / {strict}/{denominator}|"
                    f"{data['overflow']}|{json.dumps(data['retrieval_paths'], sort_keys=True)}|"
                )
    base, new, metrics = summary["modes"]["baseline"], summary["modes"]["context_management"], summary["metrics"]
    lines += ["", "## Overall", ""]
    reduction = metrics["overall_total_token_reduction_percent"]
    reduction_text = "N/A" if reduction is None else f"{reduction:.2f}%"
    lines += [
        "|Metric|Baseline|Context Management|",
        "|---|--:|--:|",
        f"|Total tokens|{base['usage']['total_tokens']}|{new['usage']['total_tokens']}|",
        f"|Token reduction|—|{reduction_text}|",
        f"|Task success|{base['quality_rates']['task_success']:.2f}%|{new['quality_rates']['task_success']:.2f}%|",
        f"|Safety compliance|{base['quality_rates']['safety_compliance']:.2f}%|{new['quality_rates']['safety_compliance']:.2f}%|",
        f"|Strict success|{base['quality_rates']['strict_success']:.2f}%|{new['quality_rates']['strict_success']:.2f}%|",
        f"|Invalid artifact calls|{base['invalid_artifact_call_count']}|{new['invalid_artifact_call_count']}|",
        f"|Overflow count|{base['context_overflow_count']}|{new['context_overflow_count']}|",
        "",
    ]
    if comparison := summary.get("optimization_comparison"):
        before, after = comparison["before"], comparison["after"]
        lines += [
            "|Metric|Before optimization|After optimization|",
            "|---|--:|--:|",
            f"|total tokens|{before['total_tokens']}|{after['total_tokens']}|",
            f"|get_tool_result calls|{before['get_tool_result_calls']}|{after['get_tool_result_calls']}|",
            f"|search_tool_result calls|{before['search_tool_result_calls']}|{after['search_tool_result_calls']}|",
            f"|artifact chars returned to model|{before['artifact_chars_returned_to_model']}|{after['artifact_chars_returned_to_model']}|",
            f"|retrieval token cost (estimated)|{before['retrieval_token_cost_estimated']}|{after['retrieval_token_cost_estimated']}|",
            f"|retrieval guard triggers|{before['retrieval_guard_trigger_count']}|{after['retrieval_guard_trigger_count']}|",
            f"|success rate|{before['success_rate']:.2f}%|{after['success_rate']:.2f}%|",
            f"|overflow count|{before['overflow_count']}|{after['overflow_count']}|",
            f"|retrieval paths|{json.dumps(before['retrieval_path_distribution'], sort_keys=True)}|{json.dumps(after['retrieval_path_distribution'], sort_keys=True)}|",
            "",
        ]
    fields = [
        ("Baseline cumulative prompt tokens", base["usage"]["prompt_tokens"]), ("New cumulative prompt tokens", new["usage"]["prompt_tokens"]),
        ("Prompt token reduction %", metrics["prompt_token_reduction_percent"]), ("Baseline cumulative total tokens", base["usage"]["total_tokens"]),
        ("Baseline cumulative completion tokens", base["usage"]["completion_tokens"]),
        ("New cumulative completion tokens", new["usage"]["completion_tokens"]),
        ("New cumulative total tokens", new["usage"]["total_tokens"]), ("Overall total token reduction %", metrics["overall_total_token_reduction_percent"]),
        ("Mean per-case reduction %", metrics["mean_per_case_reduction_percent"]), ("Median per-case reduction %", metrics["median_per_case_reduction_percent"]),
        ("Baseline task success rate", base["quality_rates"]["task_success"]), ("New task success rate", new["quality_rates"]["task_success"]),
        ("Baseline/New safety compliance", f"{base['quality_rates']['safety_compliance']:.2f}% / {new['quality_rates']['safety_compliance']:.2f}%"),
        ("Baseline/New strict success", f"{base['quality_rates']['strict_success']:.2f}% / {new['quality_rates']['strict_success']:.2f}%"),
        ("Baseline/New invalid artifact calls", f"{base['invalid_artifact_call_count']} / {new['invalid_artifact_call_count']}"),
        ("Baseline safety violations", json.dumps(base["safety_violations"], sort_keys=True)),
        ("New safety violations", json.dumps(new["safety_violations"], sort_keys=True)),
        ("Baseline/New context overflow", f"{base['context_overflow_count']} / {new['context_overflow_count']}"),
        ("Baseline/New peak prompt tokens", f"{base['peak_prompt_tokens']} / {new['peak_prompt_tokens']}"),
        ("Baseline/New model calls", f"{base['model_call_count']} / {new['model_call_count']}"),
        ("New compaction count", new["compaction_count"]), ("New rolling-summary total tokens", new["rolling_summary_usage"]["total_tokens"]),
        ("Baseline/New early constraint retention", f"{base['quality_rates']['early_constraint_retention']:.2f}% / {new['quality_rates']['early_constraint_retention']:.2f}%"),
        ("Baseline/New key fact retention", f"{base['quality_rates']['key_fact_retention']:.2f}% / {new['quality_rates']['key_fact_retention']:.2f}%"),
        ("Baseline/New Tool Result retrieval correctness", f"{base['quality_rates']['tool_result_retrieval_correctness']:.2f}% / {new['quality_rates']['tool_result_retrieval_correctness']:.2f}%"),
        ("Baseline/New retrieval protocol match", f"{base['quality_rates']['retrieval_protocol_match']:.2f}% / {new['quality_rates']['retrieval_protocol_match']:.2f}%"),
        ("Baseline/New retrieval safety", f"{base['quality_rates']['retrieval_safety_ok']:.2f}% / {new['quality_rates']['retrieval_safety_ok']:.2f}%"),
        ("Baseline/New hallucination rate", f"{base['hallucination_rate']:.2f}% / {new['hallucination_rate']:.2f}%"),
        ("Baseline/New offloaded artifacts", f"{base['offloaded_artifacts']} / {new['offloaded_artifacts']}"),
        ("Baseline/New artifact retrieval calls", f"{base['artifact_retrieval_count']} / {new['artifact_retrieval_count']}"),
        ("Baseline/New get_tool_result calls", f"{base['get_tool_result_calls']} / {new['get_tool_result_calls']}"),
        ("Baseline/New search_tool_result calls", f"{base['search_tool_result_calls']} / {new['search_tool_result_calls']}"),
        ("Baseline/New artifact chars returned to model", f"{base['artifact_chars_returned_to_model']} / {new['artifact_chars_returned_to_model']}"),
        ("Baseline/New artifact chars carried across model prompts", f"{base['artifact_context_chars_across_model_calls']} / {new['artifact_context_chars_across_model_calls']}"),
        ("Baseline/New retrieval token cost (estimated)", f"{base['retrieval_token_cost_estimated']} / {new['retrieval_token_cost_estimated']}"),
        ("Baseline/New retrieval prompt tokens carried (estimated)", f"{base['retrieval_prompt_tokens_carried_estimated']} / {new['retrieval_prompt_tokens_carried_estimated']}"),
        ("Baseline/New retrieval guard triggers", f"{base['retrieval_guard_trigger_count']} / {new['retrieval_guard_trigger_count']}"),
        ("Baseline/New retrieval errors", f"{base['retrieval_error_count']} / {new['retrieval_error_count']}"),
        ("Baseline retrieval path distribution", json.dumps(base["retrieval_path_distribution"], sort_keys=True)),
        ("New retrieval path distribution", json.dumps(new["retrieval_path_distribution"], sort_keys=True)),
    ]
    for label, value in fields:
        rendered = f"{value:.2f}%" if isinstance(value, float) else ("N/A" if value is None else str(value))
        lines.append(f"- {label}: {rendered}")
    if summary["conclusion_available"]:
        overall = metrics["overall_total_token_reduction_percent"]
        mean = metrics["mean_per_case_reduction_percent"]
        overall_text = f"降低 {overall:.2f}%" if overall >= 0 else f"增加 {-overall:.2f}%"
        mean_text = f"平均降低 {mean:.2f}%" if mean >= 0 else f"平均增加 {-mean:.2f}%"
        suite_label = "长上下文" if summary["suite"] == "long-context" else "大型 Tool Result 专项"
        lines += ["", f"在本次{suite_label} benchmark 中，与原始 baseline Agent 相比，Context Management 版本整体 total token 消耗{overall_text}；各 case token 消耗{mean_text}，同时任务成功率由 {base['quality_rates']['task_success']:.2f}% 变为 {new['quality_rates']['task_success']:.2f}%。"]
        lines += [
            "Task success 表示任务答案是否正确完成；Safety compliance 表示 artifact retrieval 行为是否安全合规；"
            "Strict success 仅在前两项同时满足时成立，且不替代 task success。"
        ]
        search_only = new["retrieval_path_distribution"].get("search_only", 0)
        if search_only:
            lines.append(
                f"其中 {search_only} 个 Context Management run 采用 search-only retrieval；"
                "该路径不影响 task success，但可能不匹配兼容性工具链指标。"
            )
    else:
        lines += ["", "无法生成总体结论：存在不完整配对、usage 缺失或控制变量违规。", "", "```json", json.dumps(summary["issues"], ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines) + "\n"


def _artifact_chars_for_row(row: dict[str, Any]) -> int:
    """Read the new telemetry, or derive legacy fixed-fixture page sizes."""
    if "artifact_chars_returned_to_model" in row:
        return int(row["artifact_chars_returned_to_model"])
    if "suite" not in row or "case_id" not in row:
        return sum(
            int(event.get("returned_artifact_chars", 0))
            for event in _retrieval_events(row.get("tool_events", []))
        )
    case = next((item for item in cases_for(row["suite"]) if item.id == row["case_id"]), None)
    total_chars = len(fixture_payload(case.fixture_kind)) if case and case.fixture_kind else 0
    returned = 0
    for event in row.get("tool_events", []):
        if event.get("name") != "get_tool_result" or event.get("status") != "ok":
            continue
        arguments = event.get("arguments", {})
        try:
            offset = int(arguments.get("offset", 0))
            limit = int(arguments.get("limit", 4096))
        except (TypeError, ValueError):
            continue
        returned += max(0, min(limit, total_chars - offset))
    return returned


def _optimization_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if rows:
        rows = _rescore_rows(rows, str(rows[0].get("suite", "long-context")))
    selected = [
        row for row in rows
        if row.get("mode") == "context_management" and row.get("usage_complete")
    ]
    tool_events = [event for row in selected for event in row.get("tool_events", [])]
    return {
        "total_tokens": sum(int(row["usage"]["total_tokens"]) for row in selected),
        "get_tool_result_calls": sum(event.get("name") == "get_tool_result" for event in tool_events),
        "search_tool_result_calls": sum(event.get("name") == "search_tool_result" for event in tool_events),
        "artifact_chars_returned_to_model": sum(_artifact_chars_for_row(row) for row in selected),
        "success_rate": (
            sum(bool(row.get("quality", {}).get("task_success")) for row in selected)
            / len(selected) * 100 if selected else 0.0
        ),
        "overflow_count": sum(int(row.get("context_overflow_count", 0)) for row in selected),
        "retrieval_path_distribution": dict(sorted(Counter(
            row.get("retrieval_path", "none") for row in selected
        ).items())),
        "retrieval_token_cost_estimated": sum(
            int(row.get("retrieval_token_cost_estimated", 0)) for row in selected
        ),
        "retrieval_guard_trigger_count": sum(
            int(row.get("retrieval_guard_trigger_count", 0)) for row in selected
        ),
    }


async def run(args: argparse.Namespace) -> int:
    config = load_config()
    configured = _make_provider(config)
    defaults = config.agents.defaults.context_management
    settings = {
        "provider": config.get_provider_name(config.agents.defaults.model),
        "model": configured.get_default_model(),
        "max_iterations": 12,
        "tool_summary_max_chars": (
            args.tool_summary_max_chars
            if args.tool_summary_max_chars is not None else defaults.tool_summary_max_chars
        ),
        "artifact_search_total_snippet_chars": (
            args.artifact_search_total_snippet_chars
            if args.artifact_search_total_snippet_chars is not None
            else defaults.artifact_search_total_snippet_chars
        ),
    }
    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-ab-{args.suite}-{uuid.uuid4().hex[:8]}"
    output = Path(args.output or RESULTS_ROOT / run_id)
    output.mkdir(parents=True, exist_ok=args.resume)
    result_path = output / "results.jsonl"
    rows: list[dict[str, Any]] = []
    if args.resume and result_path.exists():
        rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = {(row["case_id"], row["mode"], row["repetition"]) for row in rows}
    provider = _make_provider(config)
    provider.generation = GenerationSettings(temperature=0, max_tokens=8192, reasoning_effort=None)
    with result_path.open("a", encoding="utf-8") as stream:
        for repetition in range(args.repetitions):
            order_modes: tuple[Mode, Mode] = ("baseline", "context_management") if repetition % 2 == 0 else ("context_management", "baseline")
            for case in cases_for(args.suite):
                for order, mode in enumerate(order_modes):
                    if (case.id, mode, repetition) in completed:
                        print(f"SKIP completed {case.id} repetition={repetition + 1}/{args.repetitions} mode={mode}", flush=True)
                        continue
                    print(f"RUN {case.id} repetition={repetition + 1}/{args.repetitions} mode={mode}", flush=True)
                    try:
                        row = await run_once(case, mode, repetition, order, settings, provider)
                    except BaseException as exc:
                        row = {
                            "schema_version": "litebot-context-ab-result/v1", "case_id": case.id, "suite": case.suite,
                            "mode": mode, "repetition": repetition, "order": order, "provider": settings["provider"], "model": settings["model"],
                            "temperature": 0, "max_tokens": 8192, "context_window_tokens": 16_384, "max_iterations": settings["max_iterations"],
                            "tool_summary_max_chars": settings["tool_summary_max_chars"],
                            "artifact_search_total_snippet_chars": settings["artifact_search_total_snippet_chars"],
                            "usage": dict(EMPTY_USAGE), "usage_complete": False, "error": f"{type(exc).__name__}: {exc}",
                            "request_signatures": [], "control_tool_events": [], "tool_events": [], "quality": {"task_success": False},
                        }
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
    rows = _rescore_rows(rows, args.suite)
    rewritten = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_write_text(result_path, rewritten)
    summary = aggregate(rows, args.suite, args.repetitions)
    summary["run_id"] = run_id
    summary["provider"] = settings["provider"]
    summary["model"] = settings["model"]
    summary["tool_summary_max_chars"] = settings["tool_summary_max_chars"]
    summary["artifact_search_total_snippet_chars"] = settings["artifact_search_total_snippet_chars"]
    before_path = Path(args.before_results) if args.before_results else (
        RESULTS_ROOT / "ab-large-tool-result-final-v3" / "results.jsonl"
        if args.suite == "large-tool-result" else None
    )
    if before_path and before_path.is_file():
        before_rows = [
            json.loads(line) for line in before_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary["optimization_comparison"] = {
            "before_results": str(before_path),
            "before": _optimization_stats(before_rows),
            "after": _optimization_stats(rows),
        }
    _atomic_write_text(output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(output / "report.md", render_markdown(summary))
    print(render_markdown(summary))
    print(f"Results: {output}")
    return 0 if summary["conclusion_available"] else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Strict Context Management A/B benchmark")
    result.add_argument("--suite", choices=SUITES, default="long-context")
    result.add_argument("--repetitions", type=int, default=3)
    result.add_argument("--output")
    result.add_argument("--run-id")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--before-results", help="Optional prior results.jsonl for optimization comparison")
    result.add_argument(
        "--tool-summary-max-chars", type=int,
        help="Explicit offloaded Tool Result preview size in characters",
    )
    result.add_argument(
        "--artifact-search-total-snippet-chars", type=int,
        help="Maximum total snippet characters returned by one artifact search",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.repetitions <= 0:
        parser().error("--repetitions must be positive")
    if args.tool_summary_max_chars is not None and args.tool_summary_max_chars <= 0:
        parser().error("--tool-summary-max-chars must be positive")
    if (
        args.artifact_search_total_snippet_chars is not None
        and args.artifact_search_total_snippet_chars <= 0
    ):
        parser().error("--artifact-search-total-snippet-chars must be positive")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
