import pytest

from benchmarks.ab_context import (
    _optimization_stats,
    aggregate,
    cases_for,
    render_markdown,
    validate_pair,
)


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
    assert "Overall total token reduction %: 27.50%" in render_markdown(summary)


def test_aggregate_refuses_incomplete_pairs(monkeypatch):
    case = cases_for("long-context")[:1]
    monkeypatch.setattr("benchmarks.ab_context.cases_for", lambda suite: case)
    summary = aggregate([_row(case[0].id, "baseline", 0, 100, 80)], "long-context", 1)
    assert summary["conclusion_available"] is False
    assert summary["metrics"]["overall_total_token_reduction_percent"] is None


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
    }

    summary = {
        "suite": "large-tool-result",
        "cases": [],
        "modes": {
            "baseline": {
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "quality_rates": {"task_success": 100.0, "early_constraint_retention": 100.0,
                                  "key_fact_retention": 100.0, "tool_result_retrieval_correctness": 100.0},
                "context_overflow_count": 0, "peak_prompt_tokens": 1, "model_call_count": 1,
                "compaction_count": 0, "rolling_summary_usage": {"total_tokens": 0},
                "hallucination_rate": 0.0, "offloaded_artifacts": 0, "artifact_retrieval_count": 0,
                "get_tool_result_calls": 0, "search_tool_result_calls": 0,
                "artifact_chars_returned_to_model": 0, "artifact_context_chars_across_model_calls": 0,
            },
            "context_management": {
                "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
                "quality_rates": {"task_success": 100.0, "early_constraint_retention": 100.0,
                                  "key_fact_retention": 100.0, "tool_result_retrieval_correctness": 100.0},
                "context_overflow_count": 0, "peak_prompt_tokens": 40, "model_call_count": 2,
                "compaction_count": 0, "rolling_summary_usage": {"total_tokens": 0},
                "hallucination_rate": 0.0, "offloaded_artifacts": 1, "artifact_retrieval_count": 2,
                "get_tool_result_calls": 1, "search_tool_result_calls": 1,
                "artifact_chars_returned_to_model": 600, "artifact_context_chars_across_model_calls": 600,
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
