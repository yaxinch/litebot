from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .loader import CaseRegistry
from .models import EvalCase, EvalResult
from .observer import EvaluationObserver
from .probes import run_probe
from .verifier import VerifierRegistry

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def _git_info() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", True


def _normalized_trace(records: list[dict[str, Any]], tool: str = "fixture") -> list[dict[str, Any]]:
    if records and all("stage" in record and "data" in record for record in records):
        return records
    trace: list[dict[str, Any]] = []
    for index, record in enumerate(records or [{}], 1):
        name = record.get("tool", record.get("name", tool))
        action = record.get("policy_action", "allow")
        status = "denied" if action == "deny" else "failed" if record.get("outcome") == "failed" else "success"
        trace.extend([
            {"tool": name, "stage": "request", "attempt": index, "round": index, "data": {}},
            {"tool": name, "stage": "policy_decision", "attempt": index, "round": index, "data": {"action": action, "reason_codes": record.get("reason_codes", []), "rule_ids": record.get("rule_ids", [])}},
            {"tool": name, "stage": "arguments", "attempt": index, "round": index, "data": {"arguments": record.get("arguments", {})}},
        ])
        if action != "deny":
            trace.extend([{"tool": name, "stage": "execution_start", "attempt": index, "round": index, "data": {}}, {"tool": name, "stage": "execution_result", "attempt": index, "round": index, "data": {"outcome": record.get("outcome", "success")}}])
        trace.append({"tool": name, "stage": "final_status", "attempt": index, "round": index, "data": {"status": status}})
    return trace


async def _execute(case: EvalCase, workspace: Path, artifact_root: Path) -> dict[str, Any]:
    if legacy_id := case.fixture.get("legacy_case"):
        from benchmarks.models import load_cases
        from benchmarks.run import CASES_DIR, run_case
        suite = "integration" if str(legacy_id).startswith("int_") else "deterministic"
        if suite == "integration":
            return await run_probe(case, workspace)
        legacy = next(item for item in load_cases(CASES_DIR / f"{suite}.json", suite) if item.id == legacy_id)
        old = await run_case(legacy, f"v2-{uuid.uuid4().hex[:8]}", artifact_root, False)
        return {
            "final_output": f"{'PASS' if old['status'] == 'PASSED' else 'FAIL'}:{case.case_id}",
            "usage": old.get("usage", {}).get("combined", {}),
            "tool_trace": _normalized_trace(old.get("tool_events", [])),
            "tool_rounds": old.get("tool_rounds", 0), "tool_calls": old.get("tool_calls", 0),
            "stop_reason": old.get("stop_reason", "completed"), "legacy": old,
        }
    if safety_id := case.fixture.get("tool_safety_case"):
        from benchmarks.tool_safety import CASES
        outcome = await CASES[str(safety_id)]()
        return {
            "final_output": f"{'PASS' if outcome.strict_success else 'FAIL'}:{case.case_id}",
            "tool_trace": _normalized_trace(outcome.audit_projection, str(safety_id)),
            "tool_denied": int(str(safety_id) in {"explicit_deny", "confirm_reject", "duplicate_block", "approver_failure_fail_closed"}),
            "details": outcome.as_result(),
        }
    if memory_id := case.fixture.get("memory_case"):
        # The direct probe exercises the same EpisodicMemoryStore without repeating the full suite.
        scenario = {"exact_constraint": "exact_recall", "cjk_constraint": "cjk_recall", "recency_importance": "recency", "agent_memory_recall": "end_to_end_injection"}.get(str(memory_id), str(memory_id))
        case.fixture["probe"] = scenario
        return await run_probe(case, workspace)
    return await run_probe(case, workspace)


async def run_case(case: EvalCase, run_id: str, artifact_root: Path, verifier: VerifierRegistry) -> EvalResult:
    started = time.perf_counter()
    observer = EvaluationObserver()
    with tempfile.TemporaryDirectory(prefix=f"litebot-eval-{case.case_id.replace('.', '-')}-") as root:
        workspace = Path(root).resolve()
        try:
            observation = await asyncio.wait_for(_execute(case, workspace, artifact_root), timeout=case.limits.timeout_seconds)
            observer.record_usage(observation.get("usage"))
            observer.record_trace(observation.get("tool_trace", []))
            observer.context_compressions = int(observation.get("context_compressions", observer.context_compressions))
            observer.memory_hits = int(observation.get("memory_hits", observer.memory_hits))
            observer.tool_denied = int(observation.get("tool_denied", observer.tool_denied))
            observer.retry_count = int(observation.get("retry_count", observer.retry_count))
            observation.update({
                "prompt_tokens": observer.prompt_tokens, "completion_tokens": observer.completion_tokens,
                "total_tokens": observer.total_tokens, "tool_rounds": observation.get("tool_rounds", observer.tool_rounds),
                "tool_calls": observation.get("tool_calls", observer.tool_calls), "tool_trace": observer.tool_trace,
                "context_compressions": observer.context_compressions, "memory_hits": observer.memory_hits,
                "tool_denied": observer.tool_denied, "retry_count": observer.retry_count,
            })
            checks = [await verifier.verify(spec, observation, workspace) for spec in case.verifiers]
            passed = bool(checks) and all(check.passed for check in checks)
            failures = [check for check in checks if not check.passed]
            return EvalResult(
                run_id=run_id, case_id=case.case_id, category=case.category, profile=case.profile,
                passed=passed, status="passed" if passed else "failed",
                prompt_tokens=observer.prompt_tokens, completion_tokens=observer.completion_tokens,
                total_tokens=observer.total_tokens, latency=round((time.perf_counter() - started) * 1000, 3),
                tool_rounds=int(observation["tool_rounds"]), tool_calls=int(observation["tool_calls"]),
                tool_trace=observer.tool_trace, context_compressions=observer.context_compressions,
                memory_hits=observer.memory_hits, tool_denied=observer.tool_denied, retry_count=observer.retry_count,
                failure_category=failures[0].failure_category if failures else None,
                verifier_reason=(f"all {len(checks)} deterministic verifiers passed" if passed else "; ".join(item.reason for item in failures)),
                verifier_results=checks, stop_reason=str(observation.get("stop_reason", "completed")),
                audit_status=str(observation.get("audit_status", "ok")), details=observation.get("details", {}),
            )
        except asyncio.TimeoutError:
            return EvalResult(run_id=run_id, case_id=case.case_id, category=case.category, profile=case.profile, status="failed", latency=round((time.perf_counter() - started) * 1000, 3), failure_category="timeout", verifier_reason="case timeout", stop_reason="timeout")
        except Exception as exc:
            return EvalResult(run_id=run_id, case_id=case.case_id, category=case.category, profile=case.profile, status="framework_error", latency=round((time.perf_counter() - started) * 1000, 3), failure_category="framework_error", verifier_reason=f"{type(exc).__name__}: {exc}", stop_reason="framework_error", audit_status="unavailable")


async def run_suite(cases: list[EvalCase], output: Path | None = None, run_id: str | None = None) -> tuple[Path, list[EvalResult]]:
    run_id = run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    output = output or RESULTS_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output / "artifacts"
    verifier = VerifierRegistry()
    results: list[EvalResult] = []
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            result = await run_case(case, run_id, artifacts, verifier)
            results.append(result)
            stream.write(result.model_dump_json() + "\n")
            stream.flush()
            print(f"{result.status.upper():15} {case.case_id} ({result.latency:.1f} ms)")
    registry = CaseRegistry()
    manifest_hash = registry.manifest_hash(cases)
    git_commit, git_dirty = _git_info()
    manifest = {"schema_version": "litebot-eval-manifest/v1", "profile": cases[0].profile.value if cases else "unknown", "case_manifest_hash": manifest_hash, "case_ids": [case.case_id for case in cases]}
    summary = {
        "schema_version": "litebot-eval-summary/v2", "run_id": run_id,
        "profile": cases[0].profile.value if cases else "unknown", "case_manifest_hash": manifest_hash,
        "counts": {status: sum(result.status == status for result in results) for status in ("passed", "failed", "skipped", "framework_error")},
        "created_at": datetime.now(UTC).isoformat(), "python": platform.python_version(), "os": platform.platform(),
        "git_commit": git_commit, "git_dirty": git_dirty,
        "models": ["scripted-v2"], "providers": ["scripted"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, results
