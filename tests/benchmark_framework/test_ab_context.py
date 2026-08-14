import pytest

from benchmarks.ab_context import (
    _BaselineUnavailableRetrievalTool,
    _optimization_stats,
    _quality,
    _rescore_rows,
    _retrieval_diagnostics,
    _retrieval_path,
    aggregate,
    cases_for,
    render_markdown,
    validate_pair,
)
from nanobot.agent.tools.base import Tool


def _row(case_id: str, mode: str, repetition: int, total: int, prompt: int, *, success=True):
    return {
        "case_id": case_id, "suite": "long-context", "mode": mode, "repetition": repetition,
        "provider": "test", "model": "same", "temperature": 0, "max_tokens": 8192,
        "context_window_tokens": 16384, "max_iterations": 12, "usage_complete": True,
        "usage": {"prompt_tokens": prompt, "completion_tokens": total - prompt, "total_tokens": total},
        "peak_prompt_tokens": prompt, "model_call_count": 1, "compaction_count": 0,
        "rolling_summary_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "context_overflow_count": 0, "offloaded_artifacts": 0, "artifact_retrieval_count": 0,
        "quality": {"task_success": success, "early_constraint_retention": success,
                    "key_fact_retention": success, "tool_result_retrieval_correctness": success,
                    "hallucination_detected": not success},
        "request_signatures": [["same", 0, 8192, "schema"]], "control_tool_events": [],
    }


def test_validate_pair_detects_control_and_usage_violations():
    baseline = _row("case", "baseline", 0, 100, 80)
    candidate = _row("case", "context_management", 0, 50, 40)
    assert validate_pair(baseline, candidate) == []
    candidate["temperature"] = 0.1
    candidate["usage_complete"] = False
    assert validate_pair(baseline, candidate) == ["temperature_mismatch", "incomplete_provider_usage"]


def test_aggregate_weighted_mean_median_and_report(monkeypatch):
    cases = cases_for("long-context")[:2]
    monkeypatch.setattr("benchmarks.ab_context.cases_for", lambda suite: cases)
    rows = []
    for repetition in range(3):
        rows += [
            _row(cases[0].id, "baseline", repetition, 100, 80),
            _row(cases[0].id, "context_management", repetition, 50, 40),
            _row(cases[1].id, "baseline", repetition, 300, 240),
            _row(cases[1].id, "context_management", repetition, 240, 192),
        ]
    summary = aggregate(rows, "long-context", 3)
    assert summary["conclusion_available"] is True
    assert summary["metrics"]["overall_total_token_reduction_percent"] == pytest.approx(27.5)
    assert summary["metrics"]["mean_per_case_reduction_percent"] == 35
    assert summary["metrics"]["median_per_case_reduction_percent"] == 35
    assert summary["modes"]["context_management"]["retrieval_path_distribution"] == {"none": 6}
    assert "Overall total token reduction %: 27.50%" in render_markdown(summary)


def test_search_only_success_is_independent_from_protocol_match():
    case = cases_for("large-tool-result")[1]
    quality = _quality(
        case, "context_management", "PAGED-RESULT-771",
        [{"name": "search_tool_result", "status": "ok", "guard_decision": "allowed"}],
    )
    assert quality["marker_correct"] is True
    assert quality["retrieval_safety_ok"] is True
    assert quality["safety_compliance"] is True
    assert quality["strict_success"] is True
    assert quality["retrieval_protocol_match"] is False
    assert quality["tool_result_retrieval_correctness"] is False
    assert quality["task_success"] is True


@pytest.mark.parametrize(
    ("output", "events", "overflow", "finish_reason", "task_success", "safety_compliance"),
    [
        ("PAGED-RESULT-771", [], 1, "stop", False, True),
        ("Error: failed PAGED-RESULT-771", [], 0, "error", False, True),
        ("PAGED-RESULT-771 LARGE-RESULT-111", [], 0, "stop", False, True),
        ("PAGED-RESULT-771", [{"name": "search_tool_result", "status": "error_result", "detail": "Error: invalid artifact id"}], 0, "stop", True, False),
        ("PAGED-RESULT-771", [{"name": "get_tool_result", "status": "error_result", "guard_decision": "blocked", "detail": "retrieval guard blocked"}], 0, "stop", True, False),
    ],
)
def test_task_success_is_independent_from_retrieval_safety(
    output, events, overflow, finish_reason, task_success, safety_compliance,
):
    quality = _quality(
        cases_for("large-tool-result")[1], "context_management", output, events,
        overflow, finish_reason,
    )
    assert quality["task_success"] is task_success
    assert quality["safety_compliance"] is safety_compliance
    assert quality["retrieval_safety_ok"] is safety_compliance
    assert quality["strict_success"] is (task_success and safety_compliance)


@pytest.mark.parametrize(
    ("event", "violation"),
    [
        ({"name": "get_tool_result", "status": "error_result", "detail": "Error: artifact unavailable in baseline mode"}, "artifact_unavailable"),
        ({"name": "get_tool_result", "status": "error_result", "detail": "Error: artifact does not belong to this session"}, "artifact_isolation_violation"),
        ({"name": "get_tool_result", "status": "error_result", "detail": "Error: unsafe path"}, "path_safety_violation"),
        ({"name": "get_tool_result", "status": "exception", "detail": "boom"}, "retrieval_exception"),
    ],
)
def test_safety_violation_classification(event, violation):
    diagnostics = _retrieval_diagnostics({"tool_events": [event], "rounds": []})
    assert diagnostics["safety_violations"] == {violation: 1}


def test_no_match_is_safe_and_invalid_artifact_calls_are_counted():
    no_match = {
        "name": "search_tool_result", "status": "ok",
        "detail": '{"status":"no_match","matches":[]}',
    }
    invalid = {
        "name": "search_tool_result", "status": "error_result",
        "detail": "Error: invalid artifact_id",
    }
    safe = _quality(
        cases_for("large-tool-result")[1], "baseline", "PAGED-RESULT-771", [no_match],
    )
    unsafe = _quality(
        cases_for("large-tool-result")[1], "baseline", "PAGED-RESULT-771", [invalid],
    )
    assert safe["safety_compliance"] is True
    assert unsafe["task_success"] is True
    assert unsafe["safety_compliance"] is False
    assert unsafe["strict_success"] is False
    assert unsafe["invalid_artifact_call_count"] == 1


class _SchemaTool(Tool):
    @property
    def name(self):
        return "get_tool_result"

    @property
    def description(self):
        return "same"

    @property
    def parameters(self):
        return {"type": "object", "properties": {"artifact_id": {"type": "string"}}}

    async def execute(self, **kwargs):
        return kwargs


@pytest.mark.asyncio
async def test_baseline_retrieval_tool_preserves_schema_and_never_reads_artifacts():
    inner = _SchemaTool()
    tool = _BaselineUnavailableRetrievalTool(inner)
    assert tool.name == inner.name
    assert tool.description == inner.description
    assert tool.parameters == inner.parameters
    assert await tool.execute(artifact_id="not-valid") == "Error: invalid artifact id"
    assert await tool.execute(artifact_id="a" * 32) == "Error: artifact unavailable in baseline mode"


def test_retrieval_path_classification_and_diagnostics():
    search = {"name": "search_tool_result", "status": "ok", "result_size_bytes": 40}
    get = {"name": "get_tool_result", "status": "ok", "result_size_bytes": 80}
    assert _retrieval_path([]) == "none"
    assert _retrieval_path([search]) == "search_only"
    assert _retrieval_path([get]) == "get_only"
    assert _retrieval_path([search, get]) == "search_then_get"
    assert _retrieval_path([get, search]) == "get_then_search"
    sequential = [
        {**get, "artifact_id": "a", "arguments": {"offset": str(offset)}, "returned_artifact_chars": 4}
        for offset in (0, 4, 8)
    ]
    assert _retrieval_path(sequential) == "sequential_scan"
    diagnostics = _retrieval_diagnostics({
        "tool_events": [search, get],
        "rounds": [{"request": {"artifact_retrieval_chars": 20}}],
    })
    assert diagnostics["retrieval_token_cost_estimated"] == 30
    assert diagnostics["retrieval_token_cost_source"] == "byte_heuristic"
    assert diagnostics["retrieval_prompt_tokens_carried_estimated"] == 5
    assert diagnostics["retrieval_guard_trigger_count"] == 0


def test_historical_search_only_row_is_rescored_without_changing_output():
    output = "The exact marker is PAGED-RESULT-771"
    row = {
        "schema_version": "litebot-context-ab-result/v1",
        "suite": "large-tool-result", "case_id": "large_paged_retrieval",
        "mode": "context_management", "output": output,
        "context_overflow_count": 0,
        "quality": {"task_success": False},
        "tool_events": [{
            "name": "search_tool_result", "status": "ok", "guard_decision": "allowed",
            "returned_artifact_chars": 100, "result_size_bytes": 440,
        }],
        "rounds": [{"phase": "agent", "finish_reason": "stop", "request": {}}],
    }
    rescored = _rescore_rows([row], "large-tool-result")[0]
    assert rescored["output"] == output
    assert rescored["quality"]["task_success"] is True
    assert rescored["quality"]["retrieval_protocol_match"] is False
    assert rescored["retrieval_path"] == "search_only"
    assert rescored["schema_version"] == "litebot-context-ab-result/v3"


def test_aggregate_refuses_incomplete_pairs(monkeypatch):
    case = cases_for("long-context")[:1]
    monkeypatch.setattr("benchmarks.ab_context.cases_for", lambda suite: case)
    summary = aggregate([_row(case[0].id, "baseline", 0, 100, 80)], "long-context", 1)
    assert summary["conclusion_available"] is False
    assert summary["metrics"]["overall_total_token_reduction_percent"] is None


def test_aggregate_keeps_task_safety_and_strict_rates_separate(monkeypatch):
    case = cases_for("long-context")[:1]
    monkeypatch.setattr("benchmarks.ab_context.cases_for", lambda suite: case)
    baseline = _row(case[0].id, "baseline", 0, 100, 80)
    baseline["tool_events"] = [{
        "name": "search_tool_result", "status": "error_result",
        "detail": "Error: invalid artifact id",
    }]
    candidate = _row(case[0].id, "context_management", 0, 80, 60)
    summary = aggregate([baseline, candidate], "long-context", 1)
    base = summary["modes"]["baseline"]
    assert base["quality_rates"]["task_success"] == 100.0
    assert base["quality_rates"]["safety_compliance"] == 0.0
    assert base["quality_rates"]["strict_success"] == 0.0
    assert base["invalid_artifact_call_count"] == 1


def test_optimization_stats_and_report_include_retrieval_metrics():
    row = _row("large_paged_retrieval", "context_management", 0, 50, 40)
    row.update({
        "suite": "large-tool-result",
        "artifact_chars_returned_to_model": 600,
        "context_overflow_count": 0,
        "tool_events": [
            {"name": "search_tool_result"},
            {"name": "get_tool_result"},
        ],
    })
    stats = _optimization_stats([row])
    assert stats == {
        "total_tokens": 50,
        "get_tool_result_calls": 1,
        "search_tool_result_calls": 1,
        "artifact_chars_returned_to_model": 600,
        "success_rate": 100.0,
        "overflow_count": 0,
        "retrieval_path_distribution": {"search_then_get": 1},
        "retrieval_token_cost_estimated": 0,
        "retrieval_guard_trigger_count": 0,
    }

    summary = {
        "suite": "large-tool-result",
        "cases": [],
        "modes": {
            "baseline": {
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "quality_rates": {"task_success": 100.0, "early_constraint_retention": 100.0,
                                  "safety_compliance": 100.0, "strict_success": 100.0,
                                  "key_fact_retention": 100.0, "tool_result_retrieval_correctness": 100.0,
                                  "retrieval_protocol_match": 100.0, "retrieval_safety_ok": 100.0},
                "context_overflow_count": 0, "peak_prompt_tokens": 1, "model_call_count": 1,
                "compaction_count": 0, "rolling_summary_usage": {"total_tokens": 0},
                "hallucination_rate": 0.0, "offloaded_artifacts": 0, "artifact_retrieval_count": 0,
                "get_tool_result_calls": 0, "search_tool_result_calls": 0,
                "artifact_chars_returned_to_model": 0, "artifact_context_chars_across_model_calls": 0,
                "retrieval_token_cost_estimated": 0, "retrieval_prompt_tokens_carried_estimated": 0,
                "retrieval_guard_trigger_count": 0, "retrieval_error_count": 0,
                "invalid_artifact_call_count": 0, "safety_violations": {},
                "retrieval_path_distribution": {"none": 1},
            },
            "context_management": {
                "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
                "quality_rates": {"task_success": 100.0, "early_constraint_retention": 100.0,
                                  "safety_compliance": 100.0, "strict_success": 100.0,
                                  "key_fact_retention": 100.0, "tool_result_retrieval_correctness": 100.0,
                                  "retrieval_protocol_match": 100.0, "retrieval_safety_ok": 100.0},
                "context_overflow_count": 0, "peak_prompt_tokens": 40, "model_call_count": 2,
                "compaction_count": 0, "rolling_summary_usage": {"total_tokens": 0},
                "hallucination_rate": 0.0, "offloaded_artifacts": 1, "artifact_retrieval_count": 2,
                "get_tool_result_calls": 1, "search_tool_result_calls": 1,
                "artifact_chars_returned_to_model": 600, "artifact_context_chars_across_model_calls": 600,
                "retrieval_token_cost_estimated": 150, "retrieval_prompt_tokens_carried_estimated": 150,
                "retrieval_guard_trigger_count": 0, "retrieval_error_count": 0,
                "invalid_artifact_call_count": 0, "safety_violations": {},
                "retrieval_path_distribution": {"search_then_get": 1},
            },
        },
        "metrics": {"prompt_token_reduction_percent": -3900.0,
                    "overall_total_token_reduction_percent": -2400.0,
                    "mean_per_case_reduction_percent": -2400.0,
                    "median_per_case_reduction_percent": -2400.0},
        "conclusion_available": True,
        "optimization_comparison": {"before": {**stats, "total_tokens": 100}, "after": stats},
    }
    report = render_markdown(summary)
    assert "|total tokens|100|50|" in report
    assert "|search_tool_result calls|1|1|" in report
    assert "|Task success|100.00%|100.00%|" in report
    assert "|Safety compliance|100.00%|100.00%|" in report
