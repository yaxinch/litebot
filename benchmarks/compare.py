from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    source = path / "results.jsonl" if path.is_dir() else path
    return {item["case_id"]: item for line in source.read_text(encoding="utf-8").splitlines() if line.strip() for item in [json.loads(line)]}


def compare_runs(baseline: Path, candidate: Path) -> tuple[dict[str, Any], bool]:
    old, new = load_results(baseline), load_results(candidate)
    regressions, warnings, metrics = [], [], []
    for case_id in sorted(old.keys() & new.keys()):
        before, after = old[case_id], new[case_id]
        if before["status"] == "PASSED" and after["status"] == "FAILED":
            regressions.append(case_id)
        if before["status"] == "PASSED" and after["status"] == "SKIPPED":
            warnings.append({"case_id": case_id, "type": "coverage_regression"})
        metric = {"case_id": case_id}
        for key in ("duration_ms", "tool_calls", "tool_errors"):
            a, b = before.get(key, 0), after.get(key, 0)
            metric[key] = {"baseline": a, "candidate": b, "delta": b - a, "percent": None if not a else round((b - a) / a * 100, 2)}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            a = before.get("usage", {}).get("combined", {}).get(key, 0)
            b = after.get("usage", {}).get("combined", {}).get(key, 0)
            metric[key] = {"baseline": a, "candidate": b, "delta": b - a, "percent": None if not a else round((b - a) / a * 100, 2)}
        metrics.append(metric)
    report = {"regressions": regressions, "warnings": warnings, "new_cases": sorted(new.keys() - old.keys()), "missing_cases": sorted(old.keys() - new.keys()), "metrics": metrics}
    return report, bool(regressions)
