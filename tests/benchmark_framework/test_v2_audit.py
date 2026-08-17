import json

import pytest

from nanobot.config.schema import ToolPolicyConfig
from nanobot.security.tool_policy import (
    PolicyAction,
    ToolPolicyDecision,
    ToolPolicyEngine,
    ToolRequestContext,
)


@pytest.mark.asyncio
async def test_v2_audit_trace_is_complete_hashed_and_redacted(tmp_path):
    engine = ToolPolicyEngine(ToolPolicyConfig(), tmp_path)
    context = ToolRequestContext("call-1", "echo", {"token": "sk-abcdefghijklmnop", "url": "https://u:p@example.com/?password=raw"}, run_id="run", session_key="session", case_id="tool_safety.secret_redaction")
    decision = ToolPolicyDecision(PolicyAction.ALLOW, "allowed", ["local_tool.allow"], argument_hash=engine.fingerprint(context.name, context.arguments))
    record = engine.audit_record(context, decision, stage="execution", outcome="success", result_summary="token=raw-secret")
    serialized = json.dumps(record)
    assert record["schema_version"] == "litebot-tool-audit/v2"
    assert [event["stage"] for event in record["tool_trace"]] == ["request", "policy_decision", "arguments", "execution_start", "execution_result", "final_status"]
    assert "sk-abcdefghijklmnop" not in serialized and "raw-secret" not in serialized
    for previous, current in zip(record["tool_trace"], record["tool_trace"][1:]):
        assert current["previous_event_hash"] == previous["event_hash"]


@pytest.mark.asyncio
async def test_denied_trace_never_records_execution(tmp_path):
    engine = ToolPolicyEngine(ToolPolicyConfig(), tmp_path)
    context = ToolRequestContext("call-2", "exec", {"command": "shutdown now"})
    decision = await engine.evaluate(context)
    record = engine.audit_record(context, decision, stage="denied", outcome="failed")
    stages = [event["stage"] for event in record["tool_trace"]]
    assert decision.action is PolicyAction.DENY
    assert "execution_start" not in stages
    assert record["tool_trace"][-1]["data"]["status"] == "denied"
