"""Strict live-provider A/B benchmark for long-context management."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from benchmarks.collector import BenchmarkCollector, RecordingProvider, RecordingTool, add_usage
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.tool_result import GetToolResultTool
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


def _quality(case: ABCase, mode: Mode, output: str, tool_events: list[dict[str, Any]]) -> dict[str, Any]:
    positions = [output.find(marker) for marker in case.required]
    retrieval_events = [event for event in tool_events if event["name"] == "get_tool_result" and event.get("status") == "ok"]
    retrieval_ok = (
        not case.requires_retrieval
        or mode == "baseline"
        or (bool(retrieval_events) and all(marker in output for marker in case.fact_markers))
    )
    hallucination = any(marker in output for marker in case.forbidden) or any(pos < 0 for pos in positions)
    ordered = positions == sorted(positions) if case.id == "post_compaction_multistep" else True
    result = {
        "early_constraint_retention": all(marker in output for marker in case.early_markers),
        "key_fact_retention": all(marker in output for marker in case.fact_markers),
        "tool_result_retrieval_correctness": retrieval_ok,
        "hallucination_detected": hallucination,
    }
    result["task_success"] = all(marker in output for marker in case.required) and ordered and retrieval_ok and not hallucination
    return result


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
            context_management_config=ContextManagementConfig(output_reserve_tokens=2048),
        )
        registry = ToolRegistry()
        registry.register(RecordingTool(GetToolResultTool(loop.artifacts, loop.context_policy.artifact_page_size), collector))
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
        control_tools = [event for event in collector.tool_events if event["name"] != "get_tool_result"]
        overflow_count = sum("context_overflow" in event.get("actions", []) for event in context_events)
        overflow_count += sum(item.get("finish_reason") == "error" and "context" in str(outputs).lower() for item in collector.rounds)
        request_signatures = sorted({
            (item["request"]["model"], item["request"]["temperature"], item["request"]["max_tokens"], item["request"]["tool_schema_sha256"])
            for item in collector.rounds if item.get("phase") == "agent"
        })
        return {
            "schema_version": "litebot-context-ab-result/v1", "case_id": case.id, "suite": case.suite,
            "mode": mode, "repetition": repetition, "order": order, "started_at": started_at,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "provider": settings["provider"], "model": settings["model"], "temperature": 0,
            "max_tokens": 8192, "context_window_tokens": 16_384, "max_iterations": settings["max_iterations"],
            "usage": usage, "usage_complete": collector.usage_complete,
            "peak_prompt_tokens": collector.peak_prompt_tokens, "model_call_count": len(collector.rounds),
            "compaction_count": sum(bool(event.get("compacted_turns")) for event in context_events),
            "compacted_turns": sum(int(event.get("compacted_turns", 0)) for event in context_events),
            "hard_truncated_turns": sum(int(event.get("hard_truncated_turns", 0)) for event in context_events),
            "rolling_summary_usage": summary_usage, "context_overflow_count": overflow_count,
            "offloaded_artifacts": loop.context_manager.total_offloaded_artifacts,
            "artifact_retrieval_count": sum(event["name"] == "get_tool_result" for event in collector.tool_events),
            "quality": _quality(case, mode, output, collector.tool_events), "output": output,
            "request_signatures": [list(item) for item in request_signatures],
            "control_tool_events": control_tools, "tool_events": collector.tool_events,
            "context_events": context_events, "rounds": collector.rounds,
        }


def _tool_signature(row: dict[str, Any]) -> list[tuple[str, str | None, str | None]]:
    return [(event["name"], event.get("arguments_sha256"), event.get("result_sha256")) for event in row["control_tool_events"]]


def validate_pair(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for key in ("case_id", "repetition", "provider", "model", "temperature", "max_tokens", "context_window_tokens", "max_iterations"):
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
        case_rows.append({
            "case_id": case.id, "valid_repetitions": len(modes["baseline"]),
            "baseline_total_tokens": totals["baseline"], "context_management_total_tokens": totals["context_management"],
            "reduction_percent": reduction,
            "baseline_success": sum(row["quality"]["task_success"] for row in modes["baseline"]),
            "context_management_success": sum(row["quality"]["task_success"] for row in modes["context_management"]),
        })

    def mode_stats(mode: str) -> dict[str, Any]:
        selected = [row for row in valid_rows if row["mode"] == mode]
        usage = dict(EMPTY_USAGE)
        summary_usage = dict(EMPTY_USAGE)
        for row in selected:
            usage = add_usage(usage, row["usage"])
            summary_usage = add_usage(summary_usage, row["rolling_summary_usage"])
        quality_keys = ("task_success", "early_constraint_retention", "key_fact_retention", "tool_result_retrieval_correctness")
        return {
            "runs": len(selected), "usage": usage, "peak_prompt_tokens": max((row["peak_prompt_tokens"] for row in selected), default=0),
            "model_call_count": sum(row["model_call_count"] for row in selected),
            "compaction_count": sum(row["compaction_count"] for row in selected), "rolling_summary_usage": summary_usage,
            "context_overflow_count": sum(row["context_overflow_count"] for row in selected),
            "offloaded_artifacts": sum(row["offloaded_artifacts"] for row in selected),
            "artifact_retrieval_count": sum(row["artifact_retrieval_count"] for row in selected),
            "quality_rates": {key: (sum(row["quality"][key] for row in selected) / len(selected) * 100 if selected else None) for key in quality_keys},
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
        "schema_version": "litebot-context-ab-summary/v1", "suite": suite, "repetitions": repetitions,
        "conclusion_available": complete, "issues": issues, "cases": case_rows, "modes": stats, "metrics": metrics,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [f"# Context Management A/B Benchmark: {summary['suite']}", "", "|Case|Baseline Total Tokens|New Total Tokens|Reduction|Baseline Success|New Success|", "|---|--:|--:|--:|--:|--:|"]
    for row in summary["cases"]:
        reduction = "N/A" if row["reduction_percent"] is None else f"{row['reduction_percent']:.2f}%"
        denominator = row["valid_repetitions"]
        lines.append(f"|{row['case_id']}|{row['baseline_total_tokens']}|{row['context_management_total_tokens']}|{reduction}|{row['baseline_success']}/{denominator}|{row['context_management_success']}/{denominator}|")
    base, new, metrics = summary["modes"]["baseline"], summary["modes"]["context_management"], summary["metrics"]
    lines += ["", "## Overall", ""]
    fields = [
        ("Baseline cumulative prompt tokens", base["usage"]["prompt_tokens"]), ("New cumulative prompt tokens", new["usage"]["prompt_tokens"]),
        ("Prompt token reduction %", metrics["prompt_token_reduction_percent"]), ("Baseline cumulative total tokens", base["usage"]["total_tokens"]),
        ("Baseline cumulative completion tokens", base["usage"]["completion_tokens"]),
        ("New cumulative completion tokens", new["usage"]["completion_tokens"]),
        ("New cumulative total tokens", new["usage"]["total_tokens"]), ("Overall total token reduction %", metrics["overall_total_token_reduction_percent"]),
        ("Mean per-case reduction %", metrics["mean_per_case_reduction_percent"]), ("Median per-case reduction %", metrics["median_per_case_reduction_percent"]),
        ("Baseline task success rate", base["quality_rates"]["task_success"]), ("New task success rate", new["quality_rates"]["task_success"]),
        ("Baseline/New context overflow", f"{base['context_overflow_count']} / {new['context_overflow_count']}"),
        ("Baseline/New peak prompt tokens", f"{base['peak_prompt_tokens']} / {new['peak_prompt_tokens']}"),
        ("Baseline/New model calls", f"{base['model_call_count']} / {new['model_call_count']}"),
        ("New compaction count", new["compaction_count"]), ("New rolling-summary total tokens", new["rolling_summary_usage"]["total_tokens"]),
        ("Baseline/New early constraint retention", f"{base['quality_rates']['early_constraint_retention']:.2f}% / {new['quality_rates']['early_constraint_retention']:.2f}%"),
        ("Baseline/New key fact retention", f"{base['quality_rates']['key_fact_retention']:.2f}% / {new['quality_rates']['key_fact_retention']:.2f}%"),
        ("Baseline/New Tool Result retrieval correctness", f"{base['quality_rates']['tool_result_retrieval_correctness']:.2f}% / {new['quality_rates']['tool_result_retrieval_correctness']:.2f}%"),
        ("Baseline/New hallucination rate", f"{base['hallucination_rate']:.2f}% / {new['hallucination_rate']:.2f}%"),
        ("Baseline/New offloaded artifacts", f"{base['offloaded_artifacts']} / {new['offloaded_artifacts']}"),
        ("Baseline/New artifact retrieval calls", f"{base['artifact_retrieval_count']} / {new['artifact_retrieval_count']}"),
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
    else:
        lines += ["", "无法生成总体结论：存在不完整配对、usage 缺失或控制变量违规。", "", "```json", json.dumps(summary["issues"], ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace) -> int:
    config = load_config()
    configured = _make_provider(config)
    settings = {"provider": config.get_provider_name(config.agents.defaults.model), "model": configured.get_default_model(), "max_iterations": 12}
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
                            "usage": dict(EMPTY_USAGE), "usage_complete": False, "error": f"{type(exc).__name__}: {exc}",
                            "request_signatures": [], "control_tool_events": [], "tool_events": [], "quality": {"task_success": False},
                        }
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
    summary = aggregate(rows, args.suite, args.repetitions)
    summary["run_id"] = run_id
    summary["provider"] = settings["provider"]
    summary["model"] = settings["model"]
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_markdown(summary), encoding="utf-8")
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
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.repetitions <= 0:
        parser().error("--repetitions must be positive")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
