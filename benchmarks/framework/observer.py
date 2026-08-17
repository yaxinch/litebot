from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationObserver:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_rounds: int = 0
    tool_calls: int = 0
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    context_compressions: int = 0
    memory_hits: int = 0
    tool_denied: int = 0
    retry_count: int = 0
    audit_status: str = "ok"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        usage = usage or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

    def record_trace(self, events: list[dict[str, Any]]) -> None:
        self.tool_trace.extend(events)
        self.tool_calls = sum(event.get("stage") == "request" for event in self.tool_trace)
        self.tool_denied = sum(event.get("stage") == "final_status" and event.get("data", {}).get("status") in {"denied", "declined"} for event in self.tool_trace)
        self.retry_count = sum(event.get("stage") == "retry_scheduled" for event in self.tool_trace)
        rounds = {event.get("round") for event in self.tool_trace if event.get("stage") == "request" and event.get("round") is not None}
        self.tool_rounds = len(rounds) or self.tool_calls

    def record_context(self, payload: dict[str, Any]) -> None:
        actions = payload.get("actions", [])
        self.context_compressions += actions.count("compacted")
        self.memory_hits += int(payload.get("episodic_entries", 0) or 0)
