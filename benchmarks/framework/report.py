from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from .models import REPORT_SCHEMA

DEFAULT_THRESHOLDS = {
    "latency": {"absolute": 25.0, "percent": 20.0},
    "tokens": {"absolute": 64.0, "percent": 10.0},
    "tool_rounds": {"absolute": 0.25, "percent": None},
}


def load_results(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = path / "results.jsonl" if path.is_dir() else path
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary_path = source.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return rows, summary


def _stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    executed = [row for row in rows if row.get("status") in {"passed", "failed"}]
    passed = sum(row.get("status") == "passed" for row in executed)
    return {
        "pass_rate": passed / len(executed) if executed else 0.0,
        "coverage": len(executed) / len(rows) if rows else 0.0,
        "avg_latency": mean(float(row.get("latency", 0)) for row in executed) if executed else 0.0,
        "avg_tokens": mean(float(row.get("total_tokens", 0)) for row in executed) if executed else 0.0,
        "avg_tool_rounds": mean(float(row.get("tool_rounds", 0)) for row in executed) if executed else 0.0,
    }


def _delta(before: float, after: float) -> dict[str, float | None]:
    change = after - before
    return {"baseline": before, "current": after, "delta": change, "delta_percent": None if before == 0 else change / before * 100}


def compare(
    baseline_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]],
    *, baseline_meta: dict[str, Any] | None = None, current_meta: dict[str, Any] | None = None,
    strict_performance: bool = False,
) -> dict[str, Any]:
    baseline_meta, current_meta = baseline_meta or {}, current_meta or {}
    old = {row["case_id"]: row for row in baseline_rows}
    new = {row["case_id"]: row for row in current_rows}
    shared = sorted(old.keys() & new.keys())
    regressions = [case_id for case_id in shared if old[case_id].get("status") == "passed" and new[case_id].get("status") in {"failed", "skipped", "framework_error"}]
    improvements = [case_id for case_id in shared if old[case_id].get("status") == "failed" and new[case_id].get("status") == "passed"]
    missing, added = sorted(old.keys() - new.keys()), sorted(new.keys() - old.keys())
    old_stats, new_stats = _stats(baseline_rows), _stats(current_rows)
    overall = {key: _delta(old_stats[key], new_stats[key]) for key in old_stats}
    metric_warnings: list[dict[str, Any]] = []
    for metric, stat_key in (("latency", "avg_latency"), ("tokens", "avg_tokens"), ("tool_rounds", "avg_tool_rounds")):
        delta = overall[stat_key]
        threshold = DEFAULT_THRESHOLDS[metric]
        if delta["delta"] >= threshold["absolute"] and (threshold["percent"] is None or (delta["delta_percent"] or 0) >= threshold["percent"]):
            metric_warnings.append({"metric": metric, **delta, "threshold": threshold})
    categories: dict[str, Any] = {}
    for category in sorted({row.get("category") for row in baseline_rows + current_rows if row.get("category")}):
        left = [row for row in baseline_rows if row.get("category") == category]
        right = [row for row in current_rows if row.get("category") == category]
        left_ids, right_ids = {row["case_id"]: row for row in left}, {row["case_id"]: row for row in right}
        category_shared = left_ids.keys() & right_ids.keys()
        categories[category] = {
            "metrics": {key: _delta(_stats(left)[key], _stats(right)[key]) for key in _stats(left)},
            "regressions": sorted(case_id for case_id in category_shared if left_ids[case_id].get("status") == "passed" and right_ids[case_id].get("status") != "passed"),
            "improvements": sorted(case_id for case_id in category_shared if left_ids[case_id].get("status") == "failed" and right_ids[case_id].get("status") == "passed"),
        }
    failure_keys = sorted({row.get("failure_category") for row in baseline_rows + current_rows if row.get("failure_category")})
    old_failures, new_failures = Counter(row.get("failure_category") for row in baseline_rows if row.get("failure_category")), Counter(row.get("failure_category") for row in current_rows if row.get("failure_category"))
    failure_distribution = {key: {"baseline": old_failures[key], "current": new_failures[key], "delta": new_failures[key] - old_failures[key]} for key in failure_keys}
    inventory_match = not missing and not added
    old_hash, new_hash = baseline_meta.get("case_manifest_hash"), current_meta.get("case_manifest_hash")
    manifest_hash_match = not old_hash or not new_hash or old_hash == new_hash
    schema_match = not baseline_meta.get("schema_version") or not current_meta.get("schema_version") or baseline_meta.get("schema_version") == current_meta.get("schema_version")
    profile_match = not baseline_meta.get("profile") or not current_meta.get("profile") or baseline_meta.get("profile") == current_meta.get("profile")
    model_match = not baseline_meta.get("models") or not current_meta.get("models") or baseline_meta.get("models") == current_meta.get("models")
    provider_match = not baseline_meta.get("providers") or not current_meta.get("providers") or baseline_meta.get("providers") == current_meta.get("providers")
    manifest_match = inventory_match and manifest_hash_match and schema_match and profile_match
    hard_failures = ([{"type": "correctness_regression", "case_id": item} for item in regressions] + [{"type": "missing_case", "case_id": item} for item in missing])
    audit_regressions = [case_id for case_id in shared if old[case_id].get("audit_status", "ok") == "ok" and new[case_id].get("audit_status", "ok") != "ok"]
    hard_failures.extend({"type": "audit_regression", "case_id": item} for item in audit_regressions)
    if not manifest_match:
        hard_failures.append({"type": "incompatible_manifest", "inventory_match": inventory_match, "manifest_hash_match": manifest_hash_match, "schema_match": schema_match, "profile_match": profile_match})
    if strict_performance:
        hard_failures.extend({"type": "performance_regression", **item} for item in metric_warnings)
    return {
        "schema_version": REPORT_SCHEMA,
        "metadata": {"generated_at": datetime.now(UTC).isoformat(), "strict_performance": strict_performance, "thresholds": DEFAULT_THRESHOLDS},
        "baseline": baseline_meta,
        "current": current_meta,
        "compatibility": {"compatible": manifest_match, "case_inventory_match": inventory_match, "manifest_hash_match": manifest_hash_match, "schema_match": schema_match, "profile_match": profile_match, "model_match": model_match, "provider_match": provider_match},
        "overall": overall,
        "failure_distribution": failure_distribution,
        "categories": categories,
        "case_changes": {"regressions": regressions, "improvements": improvements, "new_cases": added, "missing_cases": missing, "metric_warnings": metric_warnings},
        "gate": {"passed": not hard_failures, "hard_failures": hard_failures, "warnings": [] if strict_performance else metric_warnings, "policy": "strict_performance" if strict_performance else "correctness"},
    }


def markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    lines = ["# LiteBot Regression Comparison", "", f"Gate: **{'PASS' if report['gate']['passed'] else 'FAIL'}**", "", "| Metric | Baseline | Current | Delta |", "|---|---:|---:|---:|"]
    for key in ("pass_rate", "avg_latency", "avg_tokens", "avg_tool_rounds"):
        item = overall[key]
        lines.append(f"| {key} | {item['baseline']:.4f} | {item['current']:.4f} | {item['delta']:+.4f} |")
    lines.extend(["", "## Category changes", ""])
    for category, item in report["categories"].items():
        lines.append(f"- **{category}**: {len(item['regressions'])} regression(s), {len(item['improvements'])} improvement(s)")
    lines.extend(["", "## Failure distribution", ""])
    if report["failure_distribution"]:
        for category, item in report["failure_distribution"].items():
            lines.append(f"- {category}: {item['baseline']} → {item['current']} ({item['delta']:+d})")
    else:
        lines.append("- No failures")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "comparison.md").write_text(markdown(report), encoding="utf-8")
