"""Regression contracts for the deterministic Hook lifecycle benchmark."""

import pytest

from benchmarks.hooks import CASES, run_benchmark

EXPECTED_CASES = {
    "session_lifecycle",
    "single_run_order",
    "tool_run_order",
    "multi_tool_order",
    "multi_round_run",
    "cross_turn_session",
    "hook_payload",
    "multiple_hooks_order",
    "before_hook_failure",
    "after_hook_failure",
    "legacy_adapter",
    "hook_state_isolation",
    "decision_control",
    "hook_overhead",
}


def test_hook_benchmark_has_fixed_case_inventory() -> None:
    assert set(CASES) == EXPECTED_CASES


@pytest.mark.asyncio
async def test_hook_benchmark_passes_offline() -> None:
    report = await run_benchmark(repetitions=3)
    assert report["cases"] == report["passed"] == 14
    assert report["failed"] == 0
    assert report["unexpected_events"] == 0
    assert report["missing_events"] == 0
    assert report["duplicate_events"] == 0
    assert all(
        report[key] == 1.0
        for key in (
            "task_success",
            "lifecycle_correct",
            "event_order_correct",
            "payload_correct",
            "failure_policy_correct",
            "compatibility_correct",
        )
    )
