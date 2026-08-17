import json

import pytest
from pydantic import ValidationError

from benchmarks.framework.loader import CaseRegistry
from benchmarks.framework.models import EvalCase


def test_core_inventory_is_fixed_and_balanced():
    registry = CaseRegistry()
    assert registry.validate_core_inventory() == {
        "context_management": 12,
        "memory_retrieval": 12,
        "tool_safety": 12,
        "tool_reliability": 12,
        "multi_step_reasoning": 12,
        "regression": 12,
    }
    cases = registry.load("core")
    assert len({case.case_id for case in cases}) == len(cases) == 72
    assert all(not case.requirements for case in cases)
    assert all(all(verifier.type != "llm_judge" for verifier in case.verifiers) for case in cases)


def test_judge_requires_explicit_metadata():
    data = {
        "schema_version": "litebot-eval-case/v2", "case_id": "regression.judge",
        "category": "regression", "profile": "judge", "description": "judge",
        "turns": [{"content": "x"}], "verifiers": [{"type": "llm_judge"}],
    }
    with pytest.raises(ValidationError):
        EvalCase.model_validate(data)


def test_published_case_json_is_parseable():
    for path in (CaseRegistry().root).glob("*.json"):
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), list)
