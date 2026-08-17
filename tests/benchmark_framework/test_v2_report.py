from benchmarks.framework.report import compare, markdown


def _row(case_id, category, status="passed", **fields):
    return {"case_id": case_id, "category": category, "status": status, "latency": 10, "total_tokens": 100, "tool_rounds": 1, "audit_status": "ok", **fields}


def test_comparison_has_category_failure_distribution_and_gate():
    old = [_row("tool_safety.a", "tool_safety"), _row("memory_retrieval.a", "memory_retrieval", "failed", failure_category="memory_retrieval")]
    new = [_row("tool_safety.a", "tool_safety", "failed", failure_category="policy_violation"), _row("memory_retrieval.a", "memory_retrieval")]
    report = compare(old, new)
    assert report["case_changes"]["regressions"] == ["tool_safety.a"]
    assert report["case_changes"]["improvements"] == ["memory_retrieval.a"]
    assert report["failure_distribution"]["policy_violation"]["current"] == 1
    assert report["categories"]["tool_safety"]["regressions"] == ["tool_safety.a"]
    assert report["gate"]["passed"] is False
    assert "LiteBot Regression Comparison" in markdown(report)


def test_performance_is_warning_unless_strict():
    old = [_row("regression.a", "regression")]
    new = [_row("regression.a", "regression", latency=100, total_tokens=300)]
    assert compare(old, new)["gate"]["passed"] is True
    assert compare(old, new)["gate"]["warnings"]
    assert compare(old, new, strict_performance=True)["gate"]["passed"] is False
