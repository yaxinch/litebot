from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CASE_SCHEMA = "litebot-eval-case/v2"
RESULT_SCHEMA = "litebot-eval-result/v2"
REPORT_SCHEMA = "litebot-comparison-report/v2"
AUDIT_SCHEMA = "litebot-tool-audit/v2"


class Category(StrEnum):
    CONTEXT_MANAGEMENT = "context_management"
    MEMORY_RETRIEVAL = "memory_retrieval"
    TOOL_SAFETY = "tool_safety"
    TOOL_RELIABILITY = "tool_reliability"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    REGRESSION = "regression"


class Profile(StrEnum):
    CORE = "core"
    INTEGRATION = "integration"
    LIVE = "live"
    JUDGE = "judge"


class Turn(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class Limits(BaseModel):
    timeout_seconds: float = Field(default=10, gt=0, le=600)
    max_iterations: int = Field(default=8, gt=0, le=100)
    max_tool_rounds: int = Field(default=8, ge=0, le=100)


class VerifierSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    target: str = "final_output"
    expected: Any = None
    selector: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    verifiers: list["VerifierSpec"] = Field(default_factory=list)


class EvalCase(BaseModel):
    schema_version: Literal[CASE_SCHEMA] = CASE_SCHEMA
    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    category: Category
    profile: Profile = Profile.CORE
    description: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    fixture: dict[str, Any] = Field(default_factory=dict)
    turns: list[Turn]
    limits: Limits = Field(default_factory=Limits)
    verifiers: list[VerifierSpec]
    deterministic_unavailable_reason: str | None = None
    judge_model: str | None = None
    judge_prompt_version: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "EvalCase":
        prefix, _ = self.case_id.split(".", 1)
        if prefix != self.category.value:
            raise ValueError("case_id prefix must equal category")
        if not self.turns:
            raise ValueError("at least one turn is required")
        judge = any(item.type == "llm_judge" for item in self.verifiers)
        if judge and self.profile is not Profile.JUDGE:
            raise ValueError("llm_judge is only allowed in the judge profile")
        if judge and not all((self.deterministic_unavailable_reason, self.judge_model, self.judge_prompt_version)):
            raise ValueError("judge cases require reason, model, and prompt version")
        if self.profile is Profile.CORE and self.requirements:
            raise ValueError("core cases may not require external capabilities")
        return self


class VerifierResult(BaseModel):
    type: str
    passed: bool
    reason: str
    expected: Any = None
    actual_preview: Any = None
    failure_category: str | None = None


class EvalResult(BaseModel):
    schema_version: Literal[RESULT_SCHEMA] = RESULT_SCHEMA
    run_id: str
    case_id: str
    category: Category
    profile: Profile
    passed: bool = False
    status: Literal["passed", "failed", "skipped", "framework_error"] = "failed"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency: float = 0.0
    tool_rounds: int = 0
    tool_calls: int = 0
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    context_compressions: int = 0
    memory_hits: int = 0
    tool_denied: int = 0
    retry_count: int = 0
    failure_category: str | None = None
    verifier_reason: str = ""
    verifier_results: list[VerifierResult] = Field(default_factory=list)
    stop_reason: str = "completed"
    audit_status: Literal["ok", "degraded", "unavailable"] = "ok"
    details: dict[str, Any] = Field(default_factory=dict)


VerifierSpec.model_rebuild()
