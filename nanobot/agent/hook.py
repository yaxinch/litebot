"""Unified lifecycle hooks for agent, tool, context, and memory execution."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from itertools import count
from typing import Any, Awaitable, Callable, Iterable
from uuid import uuid4

from loguru import logger

from nanobot.providers.base import LLMResponse, ToolCallRequest


class LifecycleEventType(StrEnum):
    SESSION_START = "SessionStart"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    AGENT_RUN_START = "AgentRunStart"
    AGENT_RUN_END = "AgentRunEnd"
    PRE_LLM_CALL = "PreLLMCall"
    POST_LLM_CALL = "PostLLMCall"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    TOOL_ERROR = "ToolError"
    CONTEXT_COMPACT = "ContextCompact"
    MEMORY_WRITE = "MemoryWrite"
    SESSION_END = "SessionEnd"
    LLM_STREAM_DELTA = "LLMStreamDelta"
    LLM_STREAM_END = "LLMStreamEnd"
    # Compatibility-only event points. New hooks should use the public lifecycle above.
    LEGACY_BEFORE_ITERATION = "LegacyBeforeIteration"
    LEGACY_BEFORE_TOOL_BATCH = "LegacyBeforeToolBatch"
    LEGACY_AFTER_ITERATION = "LegacyAfterIteration"
    LEGACY_FINALIZE_CONTENT = "LegacyFinalizeContent"


class HookAction(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"


@dataclass(slots=True)
class HookDecision:
    action: HookAction
    reason: str | None = None
    response_content: str | None = None
    patch: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls) -> "HookDecision":
        return cls(HookAction.ALLOW)

    @classmethod
    def deny(cls, reason: str, response_content: str | None = None) -> "HookDecision":
        return cls(HookAction.DENY, reason=reason, response_content=response_content)

    @classmethod
    def modify(cls, patch: dict[str, Any]) -> "HookDecision":
        return cls(HookAction.MODIFY, patch=patch)


@dataclass(slots=True)
class HookFailure:
    handler_name: str
    stage: str
    error: str


@dataclass(slots=True)
class LifecycleEvent:
    type: LifecycleEventType
    payload: dict[str, Any] = field(default_factory=dict)
    phase: str = "before"
    run_id: str | None = None
    session_key: str | None = None
    channel: str | None = None
    chat_id: str | None = None
    message_id: str | None = None
    iteration: int | None = None
    state: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)
    occurred_at: datetime = field(default_factory=datetime.now)
    failures: list[HookFailure] = field(default_factory=list)
    decision: HookDecision | None = None


HookHandler = Callable[[LifecycleEvent], HookDecision | None | Awaitable[HookDecision | None]]
HookFilter = Callable[[LifecycleEvent], bool | Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class HookRegistration:
    id: str
    events: frozenset[LifecycleEventType]
    handler: HookHandler
    priority: int
    name: str
    filter: HookFilter | None = None
    order: int = 0


class HookManager:
    """Ordered, fault-isolated dispatch for lifecycle events."""

    def __init__(self) -> None:
        self._registrations: dict[str, HookRegistration] = {}
        self._order = count()

    def register(
        self,
        events: LifecycleEventType | str | Iterable[LifecycleEventType | str],
        handler: HookHandler,
        *,
        priority: int = 0,
        filter: HookFilter | None = None,
        name: str | None = None,
    ) -> HookRegistration:
        if isinstance(events, (LifecycleEventType, str)):
            events = [events]
        normalized = frozenset(LifecycleEventType(event) for event in events)
        if not normalized:
            raise ValueError("at least one lifecycle event is required")
        registration = HookRegistration(
            uuid4().hex,
            normalized,
            handler,
            priority,
            name or getattr(handler, "__qualname__", type(handler).__name__),
            filter,
            next(self._order),
        )
        self._registrations[registration.id] = registration
        return registration

    def unregister(self, registration: HookRegistration | str) -> bool:
        registration_id = (
            registration.id if isinstance(registration, HookRegistration) else registration
        )
        return self._registrations.pop(registration_id, None) is not None

    async def dispatch(self, event: LifecycleEvent) -> LifecycleEvent:
        registrations = sorted(
            (item for item in self._registrations.values() if event.type in item.events),
            key=lambda item: (-item.priority, item.order),
        )
        for registration in registrations:
            try:
                if registration.filter is not None:
                    matched = registration.filter(event)
                    if inspect.isawaitable(matched):
                        matched = await matched
                    if not matched:
                        continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._record_failure(event, registration, "filter", exc)
                continue
            try:
                result = registration.handler(event)
                if inspect.isawaitable(result):
                    result = await result
                if result is None:
                    continue
                if result.action is HookAction.MODIFY:
                    event.payload.update(result.patch)
                event.decision = result
                if result.action is HookAction.DENY:
                    break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._record_failure(event, registration, "handler", exc)
        return event

    @staticmethod
    def _record_failure(
        event: LifecycleEvent, registration: HookRegistration, stage: str, exc: Exception
    ) -> None:
        event.failures.append(HookFailure(registration.name, stage, f"{type(exc).__name__}: {exc}"))
        logger.exception(
            "Lifecycle hook {} {} failed for {} ({})",
            registration.name,
            stage,
            event.type,
            event.event_id,
        )


@dataclass(slots=True)
class AgentHookContext:
    """Mutable per-iteration state exposed to legacy runner hooks."""

    iteration: int
    messages: list[dict[str, Any]]
    response: LLMResponse | None = None
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    tool_events: list[dict[str, str]] = field(default_factory=list)
    final_content: str | None = None
    stop_reason: str | None = None
    error: str | None = None


class AgentHook:
    """Legacy single-hook API retained for backwards compatibility."""

    def wants_streaming(self) -> bool:
        return False

    async def before_iteration(self, context: AgentHookContext) -> None:
        pass

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        pass

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        pass

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        pass

    async def after_iteration(self, context: AgentHookContext) -> None:
        pass

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return content


class LegacyAgentHookAdapter:
    """Route the legacy ``AgentHook`` API through a ``HookManager``.

    Registrations are scoped to one run so a shared manager remains safe when
    multiple sessions execute concurrently.
    """

    def __init__(self, hook: AgentHook, run_id: str):
        self.hook = hook
        self.run_id = run_id

    def wants_streaming(self) -> bool:
        try:
            return self.hook.wants_streaming()
        except Exception:
            logger.exception("Legacy hook streaming preference failed for run {}", self.run_id)
            return False

    def install(self, manager: HookManager) -> list[HookRegistration]:
        def scoped(event: LifecycleEvent) -> bool:
            return event.run_id == self.run_id

        async def before_iteration(event: LifecycleEvent) -> None:
            await self.hook.before_iteration(event.payload["context"])

        async def stream_delta(event: LifecycleEvent) -> None:
            await self.hook.on_stream(event.payload["context"], event.payload["delta"])

        async def stream_end(event: LifecycleEvent) -> None:
            await self.hook.on_stream_end(
                event.payload["context"],
                resuming=bool(event.payload["resuming"]),
            )

        async def before_tool_batch(event: LifecycleEvent) -> None:
            await self.hook.before_execute_tools(event.payload["context"])

        async def after_iteration(event: LifecycleEvent) -> None:
            await self.hook.after_iteration(event.payload["context"])

        def finalize_content(event: LifecycleEvent) -> None:
            event.payload["content"] = self.hook.finalize_content(
                event.payload["context"],
                event.payload.get("content"),
            )

        handlers: list[tuple[LifecycleEventType, HookHandler]] = [
            (LifecycleEventType.LEGACY_BEFORE_ITERATION, before_iteration),
            (LifecycleEventType.LLM_STREAM_DELTA, stream_delta),
            (LifecycleEventType.LLM_STREAM_END, stream_end),
            (LifecycleEventType.LEGACY_BEFORE_TOOL_BATCH, before_tool_batch),
            (LifecycleEventType.LEGACY_AFTER_ITERATION, after_iteration),
            (LifecycleEventType.LEGACY_FINALIZE_CONTENT, finalize_content),
        ]
        return [
            manager.register(event_type, handler, filter=scoped, name=f"legacy:{event_type.value}")
            for event_type, handler in handlers
        ]
