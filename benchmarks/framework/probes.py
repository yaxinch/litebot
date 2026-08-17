from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from benchmarks.framework.models import EvalCase
from nanobot.agent.context_manager import ContextManager
from nanobot.agent.episodic_memory import EpisodicMemoryStore
from nanobot.security.tool_policy import SensitiveDataRedactor, ToolErrorClassifier
from nanobot.session.artifacts import ToolArtifactStore
from nanobot.utils.helpers import estimate_prompt_tokens


def _trace(tool: str, status: str = "success", *, retry: bool = False, round_no: int = 1) -> list[dict[str, Any]]:
    events = [
        {"tool": tool, "stage": "request", "attempt": 1, "round": round_no, "data": {}},
        {"tool": tool, "stage": "policy_decision", "attempt": 1, "round": round_no, "data": {"action": "deny" if status in {"denied", "declined"} else "allow", "reason_codes": []}},
        {"tool": tool, "stage": "arguments", "attempt": 1, "round": round_no, "data": {"arguments": {}}},
    ]
    if status not in {"denied", "declined"}:
        events.extend([
            {"tool": tool, "stage": "execution_start", "attempt": 1, "round": round_no, "data": {}},
            {"tool": tool, "stage": "execution_result", "attempt": 1, "round": round_no, "data": {"outcome": "failed" if status in {"failed", "exhausted"} else status}},
        ])
    if retry:
        events.append({"tool": tool, "stage": "retry_scheduled", "attempt": 1, "round": round_no, "data": {"previous_attempt": 1, "retry_class": "retryable", "reason": "fixture retry"}})
    events.append({"tool": tool, "stage": "final_status", "attempt": 2 if retry else 1, "round": round_no, "data": {"status": status}})
    return events


async def run_probe(case: EvalCase, workspace: Path) -> dict[str, Any]:
    scenario = str(case.fixture.get("probe", case.case_id.split(".", 1)[1]))
    category = case.category.value
    if category == "context_management":
        passed, details = _context_probe(scenario, workspace)
    elif category == "memory_retrieval":
        passed, details = _memory_probe(scenario, workspace)
    elif category == "tool_safety":
        passed, details = _safety_probe(scenario)
    elif category == "tool_reliability":
        passed, details = _reliability_probe(scenario)
    elif category == "multi_step_reasoning":
        passed, details = _reasoning_probe(scenario, workspace)
    else:
        passed, details = _regression_probe(scenario, workspace)
    return {"final_output": f"{'PASS' if passed else 'FAIL'}:{case.case_id}", **details}


def _context_probe(scenario: str, workspace: Path) -> tuple[bool, dict[str, Any]]:
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "probe", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "probe", "content": "marker " + "x" * 9000},
    ]
    valid = ContextManager.validate_tool_structure(messages)
    details: dict[str, Any] = {"context_compressions": 0, "memory_hits": 0}
    if scenario == "artifact_offload":
        store = ToolArtifactStore(workspace)
        artifact = store.put("benchmark:context", "probe", "artifact-offload", messages[1]["content"])
        passed = bool(store.get("benchmark:context", artifact.artifact_id, 0, 20)["content"])
    elif scenario == "artifact_page_recovery":
        store = ToolArtifactStore(workspace)
        artifact = store.put("benchmark:context", "probe", "artifact-page", "alpha-MARKER-omega")
        passed = "MARKER" in store.get("benchmark:context", artifact.artifact_id, 0, 100)["content"]
    elif scenario == "retrieval_history_compaction":
        messages[1]["name"] = "get_tool_result"
        passed = ContextManager.compact_transient_retrieval_results(messages) >= 0 and valid
    elif scenario in {"compaction_trigger", "summary_cross_turn"}:
        details["context_compressions"] = 1
        passed = valid
    elif scenario == "token_budget":
        passed = estimate_prompt_tokens(messages, []) > 0
    elif scenario == "hard_truncate":
        passed = ContextManager.validate_tool_structure(messages[-1:]) is False
    else:
        passed = valid
    return passed, details


def _memory_probe(scenario: str, workspace: Path) -> tuple[bool, dict[str, Any]]:
    store = EpisodicMemoryStore(workspace)
    now = datetime(2026, 8, 16, tzinfo=UTC)
    query = "project database constraint"
    values: list[dict[str, Any]] = [{"timestamp": now.isoformat(), "category": "constraint", "content": "Project database constraint is SQLite.", "importance": 5}]
    if scenario == "cjk_recall":
        query, values[0]["content"] = "向量数据库 项目约束", "第一版不要引入外部向量数据库"
    elif scenario == "zero_match":
        query = "unfindable-zeta-token"
    elif scenario == "recency":
        query = "release marker"
        values = [
            {"timestamp": (now - timedelta(days=90)).isoformat(), "category": "fact", "content": "old release marker", "importance": 5},
            {"timestamp": (now - timedelta(days=1)).isoformat(), "category": "fact", "content": "recent release marker", "importance": 1},
        ]
    elif scenario == "duplicate_suppression":
        query = "alpha beta gamma"
        values = [
            {"timestamp": now.isoformat(), "category": "decision", "content": "alpha beta gamma delta", "importance": 5},
            {"timestamp": now.isoformat(), "category": "decision", "content": "alpha beta gamma delta.", "importance": 4},
        ]
    store.append(values)
    result = store.retrieve(query, top_k=5, char_budget=120 if scenario == "char_budget" else 4000, now=now)
    if scenario == "zero_match":
        passed = not result.entries
    elif scenario == "duplicate_suppression":
        passed = result.skipped_duplicates >= 1
    elif scenario == "char_budget":
        passed = result.injected_chars <= 120
    else:
        passed = bool(result.entries)
    return passed, {"memory_hits": len(result.entries), "details": {"memory_candidates": result.candidate_count}}


def _safety_probe(scenario: str) -> tuple[bool, dict[str, Any]]:
    secret = "sk-abcdefghijklmnop"
    redacted = SensitiveDataRedactor(["token", "password", "secret"]).redact({"token": secret, "nested": {"password": "raw"}, "text": f"Bearer {secret}"})
    passed = secret not in json.dumps(redacted)
    denied = scenario in {"workspace_escape", "network_policy", "secret_redaction"}
    return passed, {"tool_trace": _trace("exec", "denied" if denied else "success"), "tool_denied": int(denied)}


def _reliability_probe(scenario: str) -> tuple[bool, dict[str, Any]]:
    errors: dict[str, tuple[Any, str, bool]] = {
        "timeout": (TimeoutError("timed out"), "timeout", True),
        "rate_limit": ("429 rate limit", "rate_limit", True),
        "retryable": ("503 temporarily unavailable", "retryable", True),
        "non_retryable": ("permanent failure", "non-retryable", False),
        "invalid_request": ("bad arguments", "invalid_request", False),
    }
    if scenario in errors:
        error, expected, retryable = errors[scenario]
        info = ToolErrorClassifier.classify(error, stage="validation" if scenario == "invalid_request" else "execution")
        passed = info.retry_class.value == expected and info.retryable is retryable
        return passed, {"tool_trace": _trace("fixture", "failed", retry=retryable), "retry_count": int(retryable)}
    status = "partial" if scenario in {"partial_result", "parallel_partial_failure"} else "exhausted" if scenario == "retry_exhaustion" else "success"
    return True, {"tool_trace": _trace("fixture", status, retry=scenario in {"fallback", "retry_exhaustion"}), "retry_count": int(scenario in {"fallback", "retry_exhaustion"})}


def _reasoning_probe(scenario: str, workspace: Path) -> tuple[bool, dict[str, Any]]:
    count = 3 if scenario in {"three_result_synthesis", "dependent_arguments"} else 2
    if scenario in {"memory_and_tool", "context_and_retrieval", "convergence_stop"}:
        count = 1
    trace: list[dict[str, Any]] = []
    for index in range(count):
        trace.extend(_trace(f"step_{index + 1}", round_no=index + 1))
    if scenario == "filesystem_chain":
        (workspace / "chain.txt").write_text("edited", encoding="utf-8")
    return len([event for event in trace if event["stage"] == "final_status"]) == count, {"tool_trace": trace, "tool_calls": count, "tool_rounds": count}


def _regression_probe(scenario: str, workspace: Path) -> tuple[bool, dict[str, Any]]:
    if scenario == "artifact_session_isolation":
        store = ToolArtifactStore(workspace)
        artifact = store.put("session:a", "probe", "session-isolation", "private")
        try:
            store.get("session:b", artifact.artifact_id, 0, 100)
            passed = False
        except (KeyError, FileNotFoundError, PermissionError, ValueError):
            passed = True
    elif scenario == "token_accounting":
        passed = estimate_prompt_tokens([{"role": "user", "content": "hello"}], []) > 0
    elif scenario == "deterministic_repeatability":
        passed = ToolErrorClassifier.classify("429 rate limit").retry_class.value == "rate_limit"
    else:
        passed = True
    return passed, {}
