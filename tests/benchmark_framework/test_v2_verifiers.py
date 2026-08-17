import pytest

from benchmarks.framework.models import VerifierSpec
from benchmarks.framework.verifier import VerifierRegistry


@pytest.mark.asyncio
async def test_deterministic_verifiers(tmp_path):
    (tmp_path / "answer.txt").write_text("marker", encoding="utf-8")
    observation = {
        "final_output": '{"name":"litebot","count":2}', "count": 2,
        "tool_trace": [
            {"tool": "exec", "stage": "request", "attempt": 1, "data": {}},
            {"tool": "exec", "stage": "policy_decision", "attempt": 1, "data": {"action": "deny", "reason_codes": ["rule.deny"]}},
            {"tool": "exec", "stage": "final_status", "attempt": 1, "data": {"status": "denied"}},
        ],
    }
    specs = [
        VerifierSpec(type="contains", expected="litebot"),
        VerifierSpec(type="regex", expected=r'"count":\s*2'),
        VerifierSpec(type="json_schema", expected={"type": "object", "required": ["name"]}),
        VerifierSpec(type="file_exists", target="answer.txt"),
        VerifierSpec(type="file_contains", target="answer.txt", expected="marker"),
        VerifierSpec(type="numeric_range", target="count", options={"min": 1, "max": 3}),
        VerifierSpec(type="policy_decision", selector={"tool": "exec", "attempt": 1}, expected={"action": "deny", "reason_codes": ["rule.deny"]}),
        VerifierSpec(type="tool_trace", selector={"tool": "exec"}, expected={"execution_count": 0, "final_status": "denied"}),
    ]
    registry = VerifierRegistry()
    results = [await registry.verify(spec, observation, tmp_path) for spec in specs]
    assert all(result.passed for result in results)


@pytest.mark.asyncio
async def test_unknown_and_disabled_judge_fail_closed(tmp_path):
    registry = VerifierRegistry()
    unknown = await registry.verify(VerifierSpec(type="unknown"), {}, tmp_path)
    judge = await registry.verify(VerifierSpec(type="llm_judge"), {}, tmp_path)
    assert not unknown.passed and unknown.failure_category == "framework_error"
    assert not judge.passed and judge.failure_category == "judge_error"
