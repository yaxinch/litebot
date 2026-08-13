from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CASE_SCHEMA = "litebot-benchmark-case/v1"
RESULT_SCHEMA = "litebot-benchmark-result/v1"
SUITES = {"deterministic", "integration", "live"}


@dataclass(slots=True)
class BenchmarkTurn:
    content: str


@dataclass(slots=True)
class BenchmarkAssertion:
    type: str
    value: Any
    path: str | None = None


@dataclass(slots=True)
class BenchmarkCase:
    schema_version: str
    id: str
    suite: str
    category: str
    description: str
    turns: list[BenchmarkTurn]
    provider_script: str | None = None
    fixtures: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    assertions: list[BenchmarkAssertion] = field(default_factory=list)
    max_iterations: int = 8

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkCase":
        case = cls(
            schema_version=data["schema_version"], id=data["id"], suite=data["suite"],
            category=data["category"], description=data["description"],
            turns=[BenchmarkTurn(**turn) for turn in data.get("turns", [])],
            provider_script=data.get("provider_script"),
            fixtures=list(data.get("fixtures", [])),
            requirements=list(data.get("requirements", [])),
            assertions=[BenchmarkAssertion(**item) for item in data.get("assertions", [])],
            max_iterations=int(data.get("max_iterations", 8)),
        )
        case.validate()
        return case

    def validate(self) -> None:
        if self.schema_version != CASE_SCHEMA:
            raise ValueError(f"{self.id}: unsupported schema {self.schema_version!r}")
        if self.suite not in SUITES:
            raise ValueError(f"{self.id}: invalid suite {self.suite!r}")
        if not self.id.startswith({"deterministic": "det_", "integration": "int_", "live": "live_"}[self.suite]):
            raise ValueError(f"{self.id}: id prefix does not match suite")
        if not self.turns:
            raise ValueError(f"{self.id}: at least one turn is required")


def load_cases(path: Path, suite: str) -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array")
    cases = [BenchmarkCase.from_dict(item) for item in raw]
    if any(case.suite != suite for case in cases):
        raise ValueError(f"{path}: contains a case from another suite")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate case ids")
    return cases


def jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value

