from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from jsonschema import Draft202012Validator

from .models import VerifierResult, VerifierSpec

Judge = Callable[[VerifierSpec, dict[str, Any]], Awaitable[dict[str, Any]]]


def _preview(value: Any, limit: int = 500) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return rendered[:limit]


def _target(spec: VerifierSpec, observation: dict[str, Any]) -> Any:
    value: Any = observation
    for part in spec.target.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def _result(spec: VerifierSpec, passed: bool, reason: str, actual: Any, category: str = "verifier_mismatch") -> VerifierResult:
    return VerifierResult(
        type=spec.type, passed=passed, reason=reason, expected=spec.expected,
        actual_preview=_preview(actual), failure_category=None if passed else category,
    )


class VerifierRegistry:
    def __init__(self, judge: Judge | None = None):
        self.judge = judge

    async def verify(self, spec: VerifierSpec, observation: dict[str, Any], workspace: Path) -> VerifierResult:
        kind = spec.type
        if kind in {"all", "any", "not"}:
            children = [await self.verify(child, observation, workspace) for child in spec.verifiers]
            if kind == "all":
                passed = bool(children) and all(child.passed for child in children)
            elif kind == "any":
                passed = any(child.passed for child in children)
            else:
                passed = len(children) == 1 and not children[0].passed
            return _result(spec, passed, f"{kind} composite: {sum(c.passed for c in children)}/{len(children)} passed", [c.model_dump() for c in children])

        actual = _target(spec, observation)
        if kind == "exact_match":
            left, right = actual, spec.expected
            if spec.options.get("trim") and isinstance(left, str) and isinstance(right, str):
                left, right = left.strip(), right.strip()
            if spec.options.get("case_insensitive") and isinstance(left, str) and isinstance(right, str):
                left, right = left.casefold(), right.casefold()
            return _result(spec, left == right, "values matched" if left == right else "values differ", actual)

        if kind == "contains":
            values = spec.expected if isinstance(spec.expected, list) else [spec.expected]
            haystack = actual if isinstance(actual, str) else json.dumps(actual, ensure_ascii=False, default=str)
            checks = [str(value) in haystack for value in values]
            passed = any(checks) if spec.options.get("mode") == "any" else all(checks)
            if spec.options.get("negate"):
                passed = not passed
            return _result(spec, passed, "containment condition satisfied" if passed else "containment condition failed", actual)

        if kind == "regex":
            flags = 0
            if "i" in str(spec.options.get("flags", "")):
                flags |= re.IGNORECASE
            matched = re.search(str(spec.expected), str(actual or ""), flags) is not None
            return _result(spec, matched, "regular expression matched" if matched else "regular expression did not match", actual)

        if kind in {"file_exists", "file_contains"}:
            relative = spec.options.get("path") or spec.target
            target = (workspace / str(relative)).resolve()
            try:
                target.relative_to(workspace.resolve())
            except ValueError:
                return _result(spec, False, "path escapes case workspace", str(target), "policy_violation")
            if kind == "file_exists":
                expected_type = spec.options.get("kind", "file")
                passed = target.is_dir() if expected_type == "dir" else target.is_file()
                return _result(spec, passed, "path exists" if passed else "path is missing", str(target))
            content = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
            passed = str(spec.expected) in content
            return _result(spec, passed, "file contains expected text" if passed else "file content mismatch", content)

        if kind == "json_schema":
            value = actual
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError as exc:
                    return _result(spec, False, f"invalid JSON: {exc.msg}", actual)
            errors = sorted(Draft202012Validator(spec.expected).iter_errors(value), key=lambda item: list(item.path))
            return _result(spec, not errors, "JSON schema valid" if not errors else errors[0].message, value)

        if kind == "numeric_range":
            minimum, maximum = spec.options.get("min"), spec.options.get("max")
            passed = isinstance(actual, (int, float)) and (minimum is None or actual >= minimum) and (maximum is None or actual <= maximum)
            return _result(spec, passed, "numeric range satisfied" if passed else "numeric value outside range", actual)

        if kind == "tool_trace":
            return self._tool_trace(spec, observation.get("tool_trace", []))

        if kind == "policy_decision":
            return self._policy(spec, observation.get("tool_trace", []))

        if kind == "llm_judge":
            if self.judge is None:
                return _result(spec, False, "LLM judge is not enabled", None, "judge_error")
            try:
                judged = await self.judge(spec, observation)
                passed = isinstance(judged, dict) and isinstance(judged.get("passed"), bool) and bool(judged.get("reason")) and judged["passed"]
                reason = str(judged.get("reason", "judge returned invalid result")) if isinstance(judged, dict) else "judge returned invalid result"
                return _result(spec, passed, reason, judged, "judge_error")
            except Exception as exc:
                return _result(spec, False, f"judge error: {type(exc).__name__}: {exc}", None, "judge_error")

        return _result(spec, False, f"unknown verifier type: {kind}", actual, "framework_error")

    def _tool_trace(self, spec: VerifierSpec, trace: list[dict[str, Any]]) -> VerifierResult:
        expected = spec.expected or {}
        selected = [event for event in trace if all(event.get(key) == value for key, value in spec.selector.items())]
        names = [event.get("tool") for event in selected if event.get("stage") == "request"]
        stages = [event.get("stage") for event in selected]
        final = next((event for event in reversed(selected) if event.get("stage") == "final_status"), {})
        execution_count = stages.count("execution_start")
        retry_count = stages.count("retry_scheduled")
        checks = [
            expected.get("execution_count", execution_count) == execution_count,
            expected.get("retry_count", retry_count) == retry_count,
            expected.get("final_status", final.get("data", {}).get("status")) == final.get("data", {}).get("status"),
        ]
        if "tools" in expected:
            checks.append(names == expected["tools"])
        if "stages" in expected:
            checks.append(stages == expected["stages"])
        passed = all(checks)
        return _result(spec, passed, "tool trace valid" if passed else "tool trace mismatch", {"tools": names, "stages": stages, "execution_count": execution_count, "retry_count": retry_count, "final": final}, "tool_execution")

    def _policy(self, spec: VerifierSpec, trace: list[dict[str, Any]]) -> VerifierResult:
        selected = [event for event in trace if event.get("stage") == "policy_decision" and all(event.get(key) == value for key, value in spec.selector.items())]
        actual = selected[-1].get("data", {}) if selected else {}
        expected = spec.expected or {}
        passed = bool(selected)
        for key, value in expected.items():
            if key in {"reason_codes", "rule_ids"}:
                passed = passed and all(item in actual.get(key, []) for item in value)
            else:
                passed = passed and actual.get(key) == value
        return _result(spec, passed, "policy decision matched" if passed else "policy decision mismatch", actual, "policy_violation")
