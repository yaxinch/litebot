import pytest

from benchmarks.ab_context import aggregate, cases_for, render_markdown, validate_pair


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
