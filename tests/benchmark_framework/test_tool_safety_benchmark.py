"""Regression contracts for the deterministic Tool Safety benchmark."""

from __future__ import annotations

import json
from typing import Any

import pytest

from benchmarks import tool_safety
from benchmarks.tool_safety import CASES, CaseOutcome, Check, run_benchmark

EXPECTED_CASES = {
    "safe_allow",
    "explicit_deny",
    "confirm_approve",
    "confirm_reject",
    "approver_failure_fail_closed",
    "duplicate_block",
    "same_tool_different_args",
    "duplicate_window_expired",
    "deny_audit_integrity",
    "confirm_audit_chain",
    "audit_write_failure",
    "invalid_tool_arguments",
    "partial_execution_failure",
}


def _semantic_projection(report: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(report))
    clean.pop("created_at", None)
    for result in clean["results"]:
        result["metrics"].pop("duration_ms", None)
    return clean


def test_tool_safety_benchmark_has_fixed_case_inventory() -> None:
    assert set(CASES) == EXPECTED_CASES
    assert len(CASES) == 13


@pytest.mark.asyncio
async def test_tool_safety_benchmark_passes_offline() -> None:
    report = await run_benchmark()
    assert report["cases"] == report["passed"] == 13
    assert report["failed"] == 0
    assert report["task_success_rate"] == 1.0
    assert report["audit_success_rate"] == 1.0
    assert report["strict_success_rate"] == 1.0
    assert all(result["strict_success"] for result in report["results"])


@pytest.mark.asyncio
async def test_tool_safety_benchmark_is_semantically_repeatable() -> None:
    first = await run_benchmark()
    second = await run_benchmark()
    assert _semantic_projection(first) == _semantic_projection(second)


@pytest.mark.asyncio
async def test_tool_safety_benchmark_filters_one_case() -> None:
    report = await run_benchmark(case="duplicate_block")
    assert report["cases"] == report["passed"] == 1
    assert report["results"][0]["case_name"] == "duplicate_block"
    assert report["strict_success_rate"] == 1.0


def test_tool_safety_cli_returns_nonzero_for_failed_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    async def failed_case() -> CaseOutcome:
        return CaseOutcome(
            "safe_allow",
            [
                Check("forced_task_failure", "task", False, "forced failure"),
                Check("audit_present", "audit", True),
            ],
        )

    monkeypatch.setitem(CASES, "safe_allow", failed_case)
    output = tmp_path / "failed-run"
    exit_code = tool_safety.main(["--case", "safe_allow", "--output", str(output)])
    assert exit_code == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["strict_success_rate"] == 0.0
