import asyncio
import json

import pytest

from nanobot.config.schema import ToolPolicyConfig
from nanobot.security.tool_policy import (
    PolicyAction,
    RetryClass,
    SensitiveDataRedactor,
    ToolAuditSink,
    ToolErrorClassifier,
    ToolPolicyEngine,
    ToolRequestContext,
)


def _context(name: str, arguments: dict, tmp_path, session: str = "s") -> ToolRequestContext:
    return ToolRequestContext(
        "call-1", name, arguments, workspace=tmp_path, run_id="r", session_key=session
    )


async def test_builtin_allow_unknown_confirm_and_rule_priority(tmp_path) -> None:
    config = ToolPolicyConfig.model_validate({
        "rules": [
            {"id": "low", "tools": ["exec"], "action": "confirm", "reason": "review"},
            {
                "id": "high", "tools": ["exec"], "action": "deny", "reason": "blocked",
                "priority": 10, "arguments": {"command": {"regex": ["secret"]}},
            },
        ],
        "duplicateWindowSeconds": 0,
    })
    engine = ToolPolicyEngine(config, tmp_path)

    assert (await engine.evaluate(_context("read_file", {"path": "README.md"}, tmp_path))).action is PolicyAction.ALLOW
    assert (await engine.evaluate(_context("mcp_demo", {}, tmp_path))).action is PolicyAction.CONFIRM
    decision = await engine.evaluate(_context("exec", {"command": "echo secret"}, tmp_path))
    assert decision.action is PolicyAction.DENY
    assert decision.rule_ids == ["high"]


async def test_duplicate_fingerprint_is_order_independent_and_scoped(tmp_path) -> None:
    engine = ToolPolicyEngine(ToolPolicyConfig(duplicate_window_seconds=5), tmp_path)
    first = await engine.evaluate(_context("web_search", {"query": "x", "count": 1}, tmp_path))
    second = await engine.evaluate(_context("web_search", {"count": 1, "query": "x"}, tmp_path))
    other = await engine.evaluate(
        _context("web_search", {"query": "x", "count": 1}, tmp_path, session="other")
    )
    assert first.action is PolicyAction.ALLOW
    assert second.action is PolicyAction.DENY
    assert second.reason_codes == ["duplicate.recent"]
    assert other.action is PolicyAction.ALLOW


@pytest.mark.parametrize("path,code", [
    ("../outside.txt", "path.traversal"),
    (".git/config", "path.sensitive"),
    (".env", "path.sensitive"),
])
async def test_path_governance_denies_traversal_and_sensitive_paths(
    tmp_path, path: str, code: str
) -> None:
    engine = ToolPolicyEngine(ToolPolicyConfig(duplicate_window_seconds=0), tmp_path)
    decision = await engine.evaluate(_context("read_file", {"path": path}, tmp_path))
    assert decision.action is PolicyAction.DENY
    assert code in decision.reason_codes


async def test_outside_workspace_confirms_or_denies_when_restricted(tmp_path) -> None:
    outside = str(tmp_path.parent / "outside.txt")
    normal = ToolPolicyEngine(ToolPolicyConfig(duplicate_window_seconds=0), tmp_path)
    restricted = ToolPolicyEngine(
        ToolPolicyConfig(duplicate_window_seconds=0), tmp_path, restrict_to_workspace=True
    )
    assert (await normal.evaluate(_context("read_file", {"path": outside}, tmp_path))).action is PolicyAction.CONFIRM
    assert (await restricted.evaluate(_context("read_file", {"path": outside}, tmp_path))).action is PolicyAction.DENY


async def test_extra_root_is_read_only_exception(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    skills = tmp_path / "builtin-skills"
    workspace.mkdir()
    skills.mkdir()
    engine = ToolPolicyEngine(
        ToolPolicyConfig(duplicate_window_seconds=0), workspace,
        restrict_to_workspace=True, extra_read_roots=[skills],
    )
    read = await engine.evaluate(_context("read_file", {"path": str(skills / "SKILL.md")}, workspace))
    write = await engine.evaluate(_context("write_file", {"path": str(skills / "SKILL.md")}, workspace))
    assert read.action is PolicyAction.ALLOW
    assert write.action is PolicyAction.DENY


def test_redactor_handles_nested_keys_bearer_and_url_password() -> None:
    redactor = SensitiveDataRedactor(["api_key", "password", "authorization", "token"])
    value = redactor.redact({
        "api_key": "sk-secret",
        "nested": [{"password": "hunter2"}],
        "header": "Bearer abc.def",
        "X-Subscription-Token": "subscription-secret",
        "url": "https://user:pass@example.com/path",
    })
    serialized = json.dumps(value)
    assert "sk-secret" not in serialized
    assert "hunter2" not in serialized
    assert "abc.def" not in serialized
    assert "subscription-secret" not in serialized
    assert ":pass@" not in serialized
    assert serialized.count("[REDACTED]") >= 4


@pytest.mark.parametrize("error,stage,category,retryable", [
    ("HTTP 429 rate limit", "result", RetryClass.RATE_LIMIT, True),
    (asyncio.TimeoutError(), "execution", RetryClass.TIMEOUT, True),
    ("503 temporarily unavailable", "result", RetryClass.RETRYABLE, True),
    ("bad arguments", "validation", RetryClass.INVALID_REQUEST, False),
    ("permission denied", "execution", RetryClass.NON_RETRYABLE, False),
])
def test_error_taxonomy(error, stage, category, retryable) -> None:
    info = ToolErrorClassifier.classify(error, stage=stage)
    assert info.retry_class is category
    assert info.retryable is retryable


async def test_audit_sink_is_redacted_and_must_stay_in_workspace(tmp_path) -> None:
    redactor = SensitiveDataRedactor(["token"])
    sink = ToolAuditSink(tmp_path, "logs/audit.jsonl", redactor)
    result = await sink.write({"arguments": {"token": "raw-secret"}})
    assert result.succeeded is True
    assert result.error is None
    assert result.record["arguments"]["token"] == "[REDACTED]"
    content = sink.path.read_text(encoding="utf-8")
    assert "raw-secret" not in content
    assert "[REDACTED]" in content
    with pytest.raises(ValueError):
        ToolAuditSink(tmp_path, "../audit.jsonl", redactor)


async def test_audit_sink_reports_injected_writer_failure_and_redacts_error(tmp_path) -> None:
    redactor = SensitiveDataRedactor(["token"])

    def fail(_line: str) -> None:
        raise OSError("simulated audit write failure: token=raw-secret")

    sink = ToolAuditSink(
        tmp_path, "logs/audit.jsonl", redactor, line_writer=fail
    )
    result = await sink.write({"tool": "exec", "arguments": {"token": "raw-secret"}})

    assert result.succeeded is False
    assert result.record["arguments"]["token"] == "[REDACTED]"
    assert "simulated audit write failure" in (result.error or "")
    assert "raw-secret" not in (result.error or "")
    assert not sink.path.exists()
