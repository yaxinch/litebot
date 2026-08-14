from __future__ import annotations

import hashlib
import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.providers.base import LLMProvider, LLMResponse

SECRET_KEYS = re.compile(r"api[_-]?key|token|authorization|password|secret", re.I)
RETRIEVAL_TOOLS = {"get_tool_result", "search_tool_result"}


def redact(value: Any, limit: int = 500) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if SECRET_KEYS.search(str(key)) else redact(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, limit) for item in value]
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    raw = usage or {}
    prompt = int(raw.get("prompt_tokens", 0) or 0)
    completion = int(raw.get("completion_tokens", 0) or 0)
    total = int(raw.get("total_tokens", 0) or 0) or prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}


def retrieval_chars(result: Any) -> int:
    """Count only artifact text delivered by a retrieval response."""
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
    except (TypeError, json.JSONDecodeError):
        return 0
    if not isinstance(parsed, dict):
        return 0
    if isinstance(parsed.get("content"), str):
        return len(parsed["content"])
    return sum(
        len(match.get("snippet", ""))
        for match in parsed.get("matches", [])
        if isinstance(match, dict) and isinstance(match.get("snippet", ""), str)
    )


@dataclass
class BenchmarkCollector:
    rounds: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    context_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tool_calls(self) -> int:
        return sum(len(item.get("tool_calls", [])) for item in self.rounds if item.get("phase", "agent") == "agent")

    @property
    def tool_rounds(self) -> int:
        return sum(bool(item.get("tool_calls")) for item in self.rounds if item.get("phase", "agent") == "agent")

    @property
    def tool_errors(self) -> int:
        return sum(item.get("status") != "ok" for item in self.tool_events)

    @property
    def usage(self) -> dict[str, int]:
        total = normalize_usage({})
        for item in self.rounds:
            total = add_usage(total, item.get("usage", {}))
        return total

    @property
    def peak_prompt_tokens(self) -> int:
        return max((int(item.get("usage", {}).get("prompt_tokens", 0)) for item in self.rounds), default=0)

    @property
    def usage_complete(self) -> bool:
        return bool(self.rounds) and all(
            all(key in item.get("usage", {}) and int(item["usage"][key]) >= 0
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            and int(item["usage"].get("total_tokens", 0)) > 0
            for item in self.rounds
        )


class RecordingProvider(LLMProvider):
    """Transparent provider proxy recording every LLM call."""

    def __init__(self, inner: LLMProvider, collector: BenchmarkCollector):
        super().__init__(inner.api_key, inner.api_base)
        self.inner = inner
        self.collector = collector
        self.generation = inner.generation
        self.phase = "agent"

    async def chat(self, **kwargs: Any) -> LLMResponse:
        started = time.perf_counter()
        response = await self.inner.chat(**kwargs)
        self._record(response, started, False, kwargs)
        return response

    async def chat_stream(self, **kwargs: Any) -> LLMResponse:
        started = time.perf_counter()
        response = await self.inner.chat_stream(**kwargs)
        self._record(response, started, True, kwargs)
        return response

    def _record(self, response: LLMResponse, started: float, streaming: bool, request: dict[str, Any]) -> None:
        messages = request.get("messages") or []
        phase = self.phase
        if messages and str(messages[0].get("content", "")).startswith("You compact session context"):
            phase = "rolling_summary"
        tools = request.get("tools") or []
        retrieval_messages = [
            message for message in messages if message.get("role") == "tool"
            and message.get("name") in RETRIEVAL_TOOLS
        ]
        self.collector.rounds.append({
            "index": len(self.collector.rounds),
            "phase": phase,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "streaming": streaming,
            "usage": normalize_usage(response.usage),
            "finish_reason": response.finish_reason,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": redact(tc.arguments)} for tc in response.tool_calls],
            "request": {
                "model": request.get("model"),
                "temperature": request.get("temperature", self.generation.temperature),
                "max_tokens": request.get("max_tokens", self.generation.max_tokens),
                "tool_schema_sha256": hashlib.sha256(json.dumps(tools, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "artifact_retrieval_chars": sum(
                    retrieval_chars(message.get("content")) for message in retrieval_messages
                ),
                "artifact_retrieval_messages": len(retrieval_messages),
            },
        })

    def get_default_model(self) -> str:
        return self.inner.get_default_model()

    def estimate_prompt_tokens(self, *args: Any, **kwargs: Any):
        return self.inner.estimate_prompt_tokens(*args, **kwargs)


class RecordingTool(Tool):
    """Tool decorator preserving the native Tool interface and behavior."""

    def __init__(self, inner: Tool, collector: BenchmarkCollector):
        self.inner = inner
        self.collector = collector

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def description(self) -> str:
        return self.inner.description

    @property
    def parameters(self) -> dict[str, Any]:
        return deepcopy(self.inner.parameters)

    def set_context(self, session_key: str) -> None:
        if hasattr(self.inner, "set_context"):
            self.inner.set_context(session_key)

    async def execute(self, **kwargs: Any) -> Any:
        started = time.perf_counter()
        event = {
            "name": self.name, "arguments": redact(kwargs),
            "arguments_sha256": hashlib.sha256(json.dumps(kwargs, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        }
        try:
            result = await self.inner.execute(**kwargs)
        except BaseException as exc:
            event.update(status="exception", error_type=type(exc).__name__, detail=redact(str(exc)))
            raise
        else:
            is_error = isinstance(result, str) and result.startswith("Error")
            serialized = json.dumps(result, sort_keys=True, ensure_ascii=False) if not isinstance(result, str) else result
            event.update(
                status="error_result" if is_error else "ok", detail=redact(result),
                result_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
                result_size_bytes=len(serialized.encode()),
            )
            if self.name in RETRIEVAL_TOOLS:
                event["retrieval_kind"] = "page" if self.name == "get_tool_result" else "search"
                event["artifact_id"] = kwargs.get("artifact_id")
                event["returned_artifact_chars"] = retrieval_chars(result)
                event["guard_decision"] = "blocked" if (
                    isinstance(result, str) and "retrieval guard blocked" in result
                ) else "allowed"
                try:
                    parsed = json.loads(result) if isinstance(result, str) else result
                except (TypeError, json.JSONDecodeError):
                    parsed = None
                if isinstance(parsed, dict):
                    event["match_offsets"] = [
                        match.get("offset") for match in parsed.get("matches", [])
                        if isinstance(match, dict)
                    ]
            return result
        finally:
            event["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self.collector.tool_events.append(event)
