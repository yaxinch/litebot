import json
from typing import Any

from nanobot.agent.hook import HookManager, LifecycleEventType
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolExecutionResult, ToolRegistry
from nanobot.config.schema import ToolPolicyConfig
from nanobot.providers.base import ToolCallRequest
from nanobot.security.tool_policy import ToolAuditSink, ToolPolicyEngine


class RecordingTool(Tool):
    def __init__(self, order: list[str], result: Any = "ok"):
        self.order = order
        self.result = result

    @property
    def name(self) -> str:
        return "record"

    @property
    def description(self) -> str:
        return "record execution"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

    async def execute(self, **kwargs: Any) -> Any:
        self.order.append("execute")
        return self.result


def _spec(tmp_path, tools, engine, handler=None) -> AgentRunSpec:
    return AgentRunSpec(
        [], tools, "test", 1, run_id="run", session_key="session",
        policy_engine=engine, confirmation_handler=handler,
    )


async def test_confirm_approval_execution_hook_and_audit_order(tmp_path) -> None:
    order: list[str] = []
    tools = ToolRegistry()
    tools.register(RecordingTool(order))
    engine = ToolPolicyEngine(ToolPolicyConfig.model_validate({
        "duplicateWindowSeconds": 0,
        "rules": [{"id": "confirm", "tools": ["record"], "action": "confirm", "reason": "review"}],
    }), tmp_path)
    hooks = HookManager()
    hooks.register(LifecycleEventType.PRE_TOOL_USE, lambda _event: order.append("pre"))
    hooks.register(LifecycleEventType.POST_TOOL_USE, lambda _event: order.append("post"))
    hooks.register(LifecycleEventType.TOOL_AUDIT, lambda _event: order.append("audit"))

    async def confirm(_context, _decision) -> bool:
        order.append("confirm")
        return True

    result, event, error = await AgentRunner(object())._run_tool(
        _spec(tmp_path, tools, engine, confirm),
        ToolCallRequest("id", "record", {"value": "x"}), hooks, 0,
    )
    assert result == "ok"
    assert event["status"] == "ok"
    assert error is None
    assert order == ["pre", "audit", "confirm", "execute", "post", "audit"]
    records = [json.loads(line) for line in engine.audit.path.read_text().splitlines()]
    assert records[-1]["confirmation"] == "approved"
    assert records[-1]["outcome"] == "success"


async def test_confirm_without_handler_fails_closed_and_audits_reason(tmp_path) -> None:
    order: list[str] = []
    tools = ToolRegistry()
    tools.register(RecordingTool(order))
    engine = ToolPolicyEngine(ToolPolicyConfig.model_validate({
        "duplicateWindowSeconds": 0,
        "rules": [{"id": "confirm", "tools": ["record"], "action": "confirm", "reason": "review"}],
    }), tmp_path)
    result, event, error = await AgentRunner(object())._run_tool(
        _spec(tmp_path, tools, engine),
        ToolCallRequest("id", "record", {"value": "x"}), None, 0,
    )
    assert result.startswith("Error:")
    assert event["status"] == "error"
    assert error is None
    assert order == []
    records = [json.loads(line) for line in engine.audit.path.read_text().splitlines()]
    assert records[-1]["confirmation"] == "declined"
    assert records[-1]["policy_reason"] == "review"


async def test_validation_happens_after_policy_and_is_invalid_request(tmp_path) -> None:
    tools = ToolRegistry()
    tools.register(RecordingTool([]))
    engine = ToolPolicyEngine(ToolPolicyConfig.model_validate({
        "duplicateWindowSeconds": 0,
        "rules": [{"id": "allow", "tools": ["record"], "action": "allow", "reason": "known"}],
    }), tmp_path)
    result, event, _ = await AgentRunner(object())._run_tool(
        _spec(tmp_path, tools, engine), ToolCallRequest("id", "record", {"value": None})
    )
    assert result.startswith("Error: Invalid parameters")
    assert event["status"] == "error"
    record = json.loads(engine.audit.path.read_text().splitlines()[-1])
    assert record["stage"] == "validation"
    assert record["error"]["retry_class"] == "invalid_request"


async def test_partial_result_adds_outcome_without_changing_legacy_status(tmp_path) -> None:
    tools = ToolRegistry()
    tools.register(RecordingTool([], ToolExecutionResult.partial(
        "two of three", completed_steps=2, failed_steps=1
    )))
    engine = ToolPolicyEngine(ToolPolicyConfig.model_validate({
        "duplicateWindowSeconds": 0,
        "rules": [{"id": "allow", "tools": ["record"], "action": "allow", "reason": "known"}],
    }), tmp_path)
    result, event, _ = await AgentRunner(object())._run_tool(
        _spec(tmp_path, tools, engine), ToolCallRequest("id", "record", {"value": "x"})
    )
    assert result == "two of three"
    assert event["status"] == "ok"
    assert event["outcome"] == "partial"
    assert event["progress"] == {"completed_steps": 2, "failed_steps": 1}
    record = json.loads(engine.audit.path.read_text().splitlines()[-1])
    assert record["outcome"] == "partial"
    assert record["progress"] == {"completed_steps": 2, "failed_steps": 1}


async def test_audit_keeps_requested_and_normalized_arguments(tmp_path) -> None:
    tools = ToolRegistry()
    tools.register(RecordingTool([]))
    engine = ToolPolicyEngine(ToolPolicyConfig.model_validate({
        "duplicateWindowSeconds": 0,
        "rules": [{"id": "allow", "tools": ["record"], "action": "allow", "reason": "known"}],
    }), tmp_path)
    await AgentRunner(object())._run_tool(
        _spec(tmp_path, tools, engine), ToolCallRequest("id", "record", {"value": 7})
    )
    record = json.loads(engine.audit.path.read_text().splitlines()[-1])
    assert record["arguments"] == {"value": 7}
    assert record["normalized_arguments"] == {"value": "7"}


async def test_rejected_tool_exposes_audit_degradation_without_execution(tmp_path) -> None:
    order: list[str] = []
    tools = ToolRegistry()
    tools.register(RecordingTool(order))
    engine = ToolPolicyEngine(ToolPolicyConfig.model_validate({
        "duplicateWindowSeconds": 0,
        "rules": [{"id": "confirm", "tools": ["record"], "action": "confirm", "reason": "review"}],
    }), tmp_path)

    def fail(_line: str) -> None:
        raise OSError("simulated audit write failure")

    engine.audit = ToolAuditSink(
        tmp_path, "logs/audit.jsonl", engine.redactor, line_writer=fail
    )
    audit_events: list[dict[str, Any]] = []
    hooks = HookManager()
    hooks.register(
        LifecycleEventType.TOOL_AUDIT,
        lambda event: audit_events.append(dict(event.payload)),
    )

    async def reject(_context, _decision) -> bool:
        return False

    result, event, error = await AgentRunner(object())._run_tool(
        _spec(tmp_path, tools, engine, reject),
        ToolCallRequest("id", "record", {"value": "x"}), hooks, 0,
    )

    assert result.startswith("Error:")
    assert event["status"] == "error"
    assert event["audit_status"] == "degraded"
    assert "simulated audit write failure" in event["audit_error"]
    assert error is None
    assert order == []
    assert len(audit_events) == 2
    assert all(item["write_succeeded"] is False for item in audit_events)
    assert all(item["degraded"] is True for item in audit_events)
