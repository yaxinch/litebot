import json

from benchmarks.compare import compare_runs


def _write(path, rows):
    path.mkdir()
    (path / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_comparison_correctness_and_coverage_rules(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _write(baseline, [
        {"case_id": "a", "status": "PASSED", "duration_ms": 10, "tool_calls": 1, "tool_errors": 0},
        {"case_id": "b", "status": "PASSED", "duration_ms": 10, "tool_calls": 0, "tool_errors": 0},
    ])
    _write(candidate, [
        {"case_id": "a", "status": "FAILED", "duration_ms": 20, "tool_calls": 2, "tool_errors": 1},
        {"case_id": "b", "status": "SKIPPED", "duration_ms": 0, "tool_calls": 0, "tool_errors": 0},
    ])
    report, failed = compare_runs(baseline, candidate)
    assert failed is True
    assert report["regressions"] == ["a"]
    assert report["warnings"] == [{"case_id": "b", "type": "coverage_regression"}]

