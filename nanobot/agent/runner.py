"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Awaitable, Callable
from uuid import uuid4

from nanobot.agent.hook import (
    AgentHook,
    AgentHookContext,
    HookAction,
    HookManager,
    LegacyAgentHookAdapter,
    LifecycleEvent,
    LifecycleEventType,
)
from nanobot.agent.tools.registry import DEFAULT_TOOL_EXECUTE, ToolRegistry
from nanobot.providers.base import LLMProvider, ToolCallRequest
from nanobot.security.tool_policy import (
    PolicyAction,
    ToolAuditWriteResult,
    ToolErrorClassifier,
    ToolOutcome,
    ToolPolicyDecision,
    ToolPolicyEngine,
    ToolRequestContext,
    error_info_dict,
)
from nanobot.utils.helpers import build_assistant_message

_DEFAULT_MAX_ITERATIONS_MESSAGE = (
    "I reached the maximum number of tool call iterations ({max_iterations}) "
    "without completing the task. You can try breaking the task into smaller steps."
)
_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    hook: AgentHook | None = None
    hook_manager: HookManager | None = None
    run_id: str | None = None
    session_key: str | None = None
    event_metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False
    policy_engine: ToolPolicyEngine | None = None
    confirmation_handler: Callable[
        [ToolRequestContext, ToolPolicyDecision], Awaitable[bool]
    ] | None = None


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    run_id: str | None = None


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    @staticmethod
    def _event(
        spec: AgentRunSpec,
        event_type: LifecycleEventType,
        payload: dict[str, Any] | None = None,
        *,
        phase: str = "before",
        iteration: int | None = None,
    ) -> LifecycleEvent:
        metadata = spec.event_metadata
        return LifecycleEvent(
            event_type,
            payload=payload or {},
            phase=phase,
            run_id=spec.run_id,
            session_key=spec.session_key,
            channel=metadata.get("channel"),
            chat_id=metadata.get("chat_id"),
            message_id=metadata.get("message_id"),
            iteration=iteration,
            state=metadata,
        )

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        spec.run_id = spec.run_id or uuid4().hex
        hook_manager = spec.hook_manager or (HookManager() if spec.hook else None)
        adapter = LegacyAgentHookAdapter(spec.hook, spec.run_id) if spec.hook else None
        registrations = adapter.install(hook_manager) if adapter and hook_manager else []
        if hook_manager:
            await hook_manager.dispatch(
                self._event(
                    spec,
                    LifecycleEventType.AGENT_RUN_START,
                    {
                        "model": spec.model,
                        "provider": type(self.provider).__name__,
                        "max_iterations": spec.max_iterations,
                    },
                )
            )
        try:
            result = await self._run_inner(spec, hook_manager, adapter)
        except BaseException as exc:
            if hook_manager:
                await hook_manager.dispatch(
                    self._event(
                        spec,
                        LifecycleEventType.AGENT_RUN_END,
                        {"result": None, "stop_reason": "exception", "error": str(exc)},
                        phase="error",
                    )
                )
            raise
        else:
            if hook_manager:
                await hook_manager.dispatch(
                    self._event(
                        spec,
                        LifecycleEventType.AGENT_RUN_END,
                        {
                            "result": result,
                            "stop_reason": result.stop_reason,
                            "error": result.error,
                        },
                        phase="after",
                    )
                )
            return result
        finally:
            if hook_manager:
                for registration in registrations:
                    hook_manager.unregister(registration)

    async def _run_inner(
        self,
        spec: AgentRunSpec,
        hook_manager: HookManager | None,
        adapter: LegacyAgentHookAdapter | None,
    ) -> AgentRunResult:
        messages = list(spec.initial_messages)
        final_content: str | None = None
        tools_used: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []

        for iteration in range(spec.max_iterations):
            context = AgentHookContext(iteration=iteration, messages=messages)
            if hook_manager and adapter:
                await hook_manager.dispatch(
                    self._event(
                        spec,
                        LifecycleEventType.LEGACY_BEFORE_ITERATION,
                        {"context": context},
                        iteration=iteration,
                    )
                )
            kwargs: dict[str, Any] = {
                "messages": messages,
                "tools": spec.tools.get_definitions(),
                "model": spec.model,
            }
            if spec.temperature is not None:
                kwargs["temperature"] = spec.temperature
            if spec.max_tokens is not None:
                kwargs["max_tokens"] = spec.max_tokens
            if spec.reasoning_effort is not None:
                kwargs["reasoning_effort"] = spec.reasoning_effort

            if hook_manager:
                event = await hook_manager.dispatch(
                    LifecycleEvent(
                        LifecycleEventType.PRE_LLM_CALL,
                        payload={
                            "messages": messages,
                            "tools": kwargs["tools"],
                            "model": spec.model,
                            "kwargs": kwargs,
                        },
                        run_id=spec.run_id,
                        session_key=spec.session_key,
                        iteration=iteration,
                        state=spec.event_metadata,
                    )
                )
                if event.decision and event.decision.action is HookAction.DENY:
                    final_content = event.decision.response_content
                    error = event.decision.reason
                    stop_reason = "denied"
                    context.final_content, context.error, context.stop_reason = (
                        final_content,
                        error,
                        stop_reason,
                    )
                    if adapter:
                        await hook_manager.dispatch(
                            self._event(
                                spec,
                                LifecycleEventType.LEGACY_AFTER_ITERATION,
                                {"context": context},
                                phase="after",
                                iteration=iteration,
                            )
                        )
                    break
                messages = event.payload.get("messages", messages)
                kwargs = event.payload.get("kwargs", kwargs)
                kwargs["messages"] = messages
                kwargs["tools"] = event.payload.get("tools", kwargs.get("tools", []))

            streaming = bool(adapter and adapter.wants_streaming())
            if streaming:

                async def _stream(delta: str) -> None:
                    if hook_manager:
                        await hook_manager.dispatch(
                            LifecycleEvent(
                                LifecycleEventType.LLM_STREAM_DELTA,
                                payload={"delta": delta, "context": context},
                                phase="after",
                                run_id=spec.run_id,
                                session_key=spec.session_key,
                                iteration=iteration,
                                state=spec.event_metadata,
                            )
                        )

                response = await self.provider.chat_stream_with_retry(
                    **kwargs,
                    on_content_delta=_stream,
                )
            else:
                response = await self.provider.chat_with_retry(**kwargs)

            if hook_manager:
                event = await hook_manager.dispatch(
                    LifecycleEvent(
                        LifecycleEventType.POST_LLM_CALL,
                        payload={
                            "response": response,
                            "content": response.content,
                            "tool_calls": response.tool_calls,
                            "usage": response.usage or {},
                        },
                        phase="after",
                        run_id=spec.run_id,
                        session_key=spec.session_key,
                        iteration=iteration,
                        state=spec.event_metadata,
                    )
                )
                if event.decision and event.decision.action is HookAction.DENY:
                    final_content, error, stop_reason = (
                        event.decision.response_content,
                        event.decision.reason,
                        "denied",
                    )
                    context.final_content, context.error, context.stop_reason = (
                        final_content,
                        error,
                        stop_reason,
                    )
                    if adapter:
                        await hook_manager.dispatch(
                            self._event(
                                spec,
                                LifecycleEventType.LEGACY_AFTER_ITERATION,
                                {"context": context},
                                phase="after",
                                iteration=iteration,
                            )
                        )
                    break
                response.content = event.payload.get("content", response.content)
                response.tool_calls = event.payload.get("tool_calls", response.tool_calls)

            raw_usage = response.usage or {}
            usage = {
                "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
            }
            context.response = response
            context.usage = usage
            context.tool_calls = list(response.tool_calls)

            if response.has_tool_calls:
                if streaming:
                    if hook_manager:
                        await hook_manager.dispatch(
                            LifecycleEvent(
                                LifecycleEventType.LLM_STREAM_END,
                                payload={"context": context, "resuming": True},
                                phase="after",
                                run_id=spec.run_id,
                                session_key=spec.session_key,
                                iteration=iteration,
                                state=spec.event_metadata,
                            )
                        )

                messages.append(
                    build_assistant_message(
                        response.content or "",
                        tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    )
                )
                tools_used.extend(tc.name for tc in response.tool_calls)

                if hook_manager and adapter:
                    await hook_manager.dispatch(
                        self._event(
                            spec,
                            LifecycleEventType.LEGACY_BEFORE_TOOL_BATCH,
                            {"context": context},
                            iteration=iteration,
                        )
                    )

                results, new_events, fatal_error = await self._execute_tools(
                    spec, response.tool_calls, hook_manager, iteration
                )
                tool_events.extend(new_events)
                context.tool_results = list(results)
                context.tool_events = list(new_events)
                if fatal_error is not None:
                    error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                    stop_reason = "tool_error"
                    context.error = error
                    context.stop_reason = stop_reason
                    if adapter:
                        await hook_manager.dispatch(
                            self._event(
                                spec,
                                LifecycleEventType.LEGACY_AFTER_ITERATION,
                                {"context": context},
                                phase="after",
                                iteration=iteration,
                            )
                        )
                    break
                for tool_call, result in zip(response.tool_calls, results):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        }
                    )
                if adapter:
                    await hook_manager.dispatch(
                        self._event(
                            spec,
                            LifecycleEventType.LEGACY_AFTER_ITERATION,
                            {"context": context},
                            phase="after",
                            iteration=iteration,
                        )
                    )
                continue

            if streaming:
                if hook_manager:
                    await hook_manager.dispatch(
                        LifecycleEvent(
                            LifecycleEventType.LLM_STREAM_END,
                            payload={"context": context, "resuming": False},
                            phase="after",
                            run_id=spec.run_id,
                            session_key=spec.session_key,
                            iteration=iteration,
                            state=spec.event_metadata,
                        )
                    )

            clean = response.content
            if hook_manager and adapter:
                finalize_event = await hook_manager.dispatch(
                    self._event(
                        spec,
                        LifecycleEventType.LEGACY_FINALIZE_CONTENT,
                        {"context": context, "content": clean},
                        iteration=iteration,
                    )
                )
                clean = finalize_event.payload.get("content", clean)
            if response.finish_reason == "error":
                final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                stop_reason = "error"
                error = final_content
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                if adapter:
                    await hook_manager.dispatch(
                        self._event(
                            spec,
                            LifecycleEventType.LEGACY_AFTER_ITERATION,
                            {"context": context},
                            phase="after",
                            iteration=iteration,
                        )
                    )
                break

            messages.append(
                build_assistant_message(
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
            )
            final_content = clean
            context.final_content = final_content
            context.stop_reason = stop_reason
            if adapter:
                await hook_manager.dispatch(
                    self._event(
                        spec,
                        LifecycleEventType.LEGACY_AFTER_ITERATION,
                        {"context": context},
                        phase="after",
                        iteration=iteration,
                    )
                )
            break
        else:
            stop_reason = "max_iterations"
            template = spec.max_iterations_message or _DEFAULT_MAX_ITERATIONS_MESSAGE
            final_content = template.format(max_iterations=spec.max_iterations)

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
            run_id=spec.run_id,
        )

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
        hook_manager: HookManager | None = None,
        iteration: int | None = None,
    ) -> tuple[list[Any], list[dict[str, str]], BaseException | None]:
        if spec.concurrent_tools:
            tool_results = await asyncio.gather(
                *(
                    self._run_tool(spec, tool_call, hook_manager, iteration)
                    for tool_call in tool_calls
                )
            )
        else:
            tool_results = [
                await self._run_tool(spec, tool_call, hook_manager, iteration)
                for tool_call in tool_calls
            ]

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        for result, event, error in tool_results:
            results.append(result)
            events.append(event)
            if error is not None and fatal_error is None:
                fatal_error = error
        return results, events, fatal_error

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
        hook_manager: HookManager | None = None,
        iteration: int | None = None,
    ) -> tuple[Any, dict[str, Any], BaseException | None]:
        started = perf_counter()
        hook_denial: tuple[str, str | None] | None = None
        if hook_manager:
            event = await hook_manager.dispatch(
                LifecycleEvent(
                    LifecycleEventType.PRE_TOOL_USE,
                    payload={
                        "tool_call": tool_call,
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                    run_id=spec.run_id,
                    session_key=spec.session_key,
                    iteration=iteration,
                    state=spec.event_metadata,
                )
            )
            if event.decision and event.decision.action is HookAction.DENY:
                detail = event.decision.reason or "tool use denied"
                hook_denial = (detail, event.decision.response_content)
            tool_call.name = event.payload.get("name", tool_call.name)
            tool_call.arguments = event.payload.get("arguments", tool_call.arguments)

        tool = spec.tools.get(tool_call.name) if isinstance(spec.tools, ToolRegistry) else None
        policy_context = ToolRequestContext(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            arguments=tool_call.arguments,
            schema=tool.parameters if tool else None,
            workspace=spec.policy_engine.workspace if spec.policy_engine else None,
            restrict_to_workspace=(
                spec.policy_engine.restrict_to_workspace if spec.policy_engine else False
            ),
            run_id=spec.run_id,
            session_key=spec.session_key,
            case_id=(str(spec.event_metadata.get("case_id")) if spec.event_metadata.get("case_id") else None),
            attempt=int(spec.event_metadata.get("attempt", 1) or 1),
        )
        policy_decision: ToolPolicyDecision | None = None
        if spec.policy_engine:
            policy_decision = await spec.policy_engine.evaluate(policy_context)
            if hook_denial:
                policy_decision = ToolPolicyDecision(
                    PolicyAction.DENY,
                    hook_denial[0],
                    ["hook.denied"],
                    argument_hash=policy_decision.argument_hash,
                )
        elif hook_denial:
            return await self._reject_tool(
                spec, tool_call, hook_denial[0], hook_denial[1], hook_manager,
                iteration, policy_context, None, started,
            )

        if policy_decision and policy_decision.action is PolicyAction.CONFIRM:
            await self._audit_tool(
                spec, hook_manager, iteration, policy_context, policy_decision,
                stage="confirmation", outcome="pending", duration_ms=self._elapsed_ms(started),
            )
            approved = False
            approval_error: str | None = None
            if spec.confirmation_handler:
                try:
                    approved = bool(
                        await spec.confirmation_handler(policy_context, policy_decision)
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    approval_error = f"{type(exc).__name__}: {exc}"
                    approved = False
            policy_decision.confirmation = (
                "approved" if approved else "error" if approval_error else "declined"
            )
            if not approved:
                detail = (
                    "tool confirmation failed because the approval handler errored"
                    if approval_error
                    else "tool confirmation was not approved"
                )
                return await self._reject_tool(
                    spec, tool_call, detail, None,
                    hook_manager, iteration, policy_context, policy_decision, started,
                    approval_error=approval_error,
                )

        if policy_decision and policy_decision.action is PolicyAction.DENY:
            return await self._reject_tool(
                spec, tool_call, policy_decision.reason, hook_denial[1] if hook_denial else None,
                hook_manager, iteration, policy_context, policy_decision, started,
            )

        try:
            if (
                isinstance(spec.tools, ToolRegistry)
                and type(spec.tools).execute is DEFAULT_TOOL_EXECUTE
            ):
                detailed = await spec.tools.execute_detailed(tool_call.name, tool_call.arguments)
                result = detailed.content
                result_status = detailed.status
                result_stage = detailed.stage
                result_outcome = detailed.outcome
                result_progress = detailed.progress
                error_info = detailed.error_info
                normalized_arguments = detailed.normalized_arguments
            else:
                result = await spec.tools.execute(tool_call.name, tool_call.arguments)
                result_status = (
                    "error" if isinstance(result, str) and result.startswith("Error") else "ok"
                )
                result_stage = "result" if result_status == "error" else None
                result_outcome = (
                    ToolOutcome.FAILED if result_status == "error" else ToolOutcome.SUCCESS
                )
                result_progress = None
                normalized_arguments = tool_call.arguments
                error_info = (
                    ToolErrorClassifier.classify(result, stage=result_stage or "result")
                    if result_status == "error" else None
                )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            if hook_manager:
                await hook_manager.dispatch(
                    LifecycleEvent(
                        LifecycleEventType.TOOL_ERROR,
                        payload={
                            "tool_call": tool_call,
                            "stage": "execution",
                            "error": str(exc),
                            "exception": exc,
                        },
                        phase="error",
                        run_id=spec.run_id,
                        session_key=spec.session_key,
                        iteration=iteration,
                        state=spec.event_metadata,
                    )
                )
            error_info = ToolErrorClassifier.classify(exc, stage="execution")
            if policy_decision:
                audit_result = await self._audit_tool(
                    spec, hook_manager, iteration, policy_context, policy_decision,
                    stage="execution", outcome=ToolOutcome.FAILED.value,
                    duration_ms=self._elapsed_ms(started), error=error_info_dict(error_info),
                )
                self._mark_audit_degraded(event, audit_result)
            if spec.fail_on_tool_error:
                return f"Error: {type(exc).__name__}: {exc}", event, exc
            return f"Error: {type(exc).__name__}: {exc}", event, None

        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        event_data = {
            "name": tool_call.name,
            "status": result_status,
            "detail": detail,
        }
        if result_outcome is ToolOutcome.PARTIAL:
            event_data["outcome"] = ToolOutcome.PARTIAL.value
            if result_progress:
                event_data["progress"] = result_progress
        if hook_manager:
            event_type = (
                LifecycleEventType.TOOL_ERROR
                if event_data["status"] == "error"
                else LifecycleEventType.POST_TOOL_USE
            )
            event = await hook_manager.dispatch(
                LifecycleEvent(
                    event_type,
                    payload={
                        "tool_call": tool_call,
                        "result": result,
                        "event": event_data,
                        "stage": result_stage,
                    },
                    phase="error" if event_type is LifecycleEventType.TOOL_ERROR else "after",
                    run_id=spec.run_id,
                    session_key=spec.session_key,
                    iteration=iteration,
                    state=spec.event_metadata,
                )
            )
            result = event.payload.get("result", result)
            detail = "" if result is None else str(result)
            detail = detail.replace("\n", " ").strip()
            if not detail:
                detail = "(empty)"
            elif len(detail) > 120:
                detail = detail[:120] + "..."
            event_data["detail"] = detail
        if policy_decision:
            audit_result = await self._audit_tool(
                spec, hook_manager, iteration, policy_context, policy_decision,
                stage=result_stage or "execution",
                outcome=result_outcome.value,
                duration_ms=self._elapsed_ms(started),
                confirmation=policy_decision.confirmation,
                progress=result_progress,
                normalized_arguments=normalized_arguments,
                error=error_info_dict(error_info),
                result_summary=detail,
            )
            self._mark_audit_degraded(event_data, audit_result)
        return result, event_data, None

    async def _reject_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
        detail: str,
        response_content: str | None,
        hook_manager: HookManager | None,
        iteration: int | None,
        context: ToolRequestContext,
        decision: ToolPolicyDecision | None,
        started: float,
        approval_error: str | None = None,
    ) -> tuple[Any, dict[str, Any], BaseException | None]:
        result = response_content or f"Error: {detail}"
        if hook_manager:
            await hook_manager.dispatch(
                LifecycleEvent(
                    LifecycleEventType.TOOL_ERROR,
                    payload={
                        "tool_call": tool_call,
                        "stage": "denied",
                        "error": detail,
                        "result": result,
                    },
                    phase="error",
                    run_id=spec.run_id,
                    session_key=spec.session_key,
                    iteration=iteration,
                    state=spec.event_metadata,
                )
            )
        event_data = {"name": tool_call.name, "status": "error", "detail": detail}
        if decision:
            error_info = ToolErrorClassifier.classify(detail, stage="denied")
            audit_result = await self._audit_tool(
                spec, hook_manager, iteration, context, decision,
                stage="denied", outcome=ToolOutcome.FAILED.value,
                duration_ms=self._elapsed_ms(started),
                confirmation=decision.confirmation,
                approval_error=approval_error,
                error=error_info_dict(error_info),
                result_summary=result,
            )
            self._mark_audit_degraded(event_data, audit_result)
        return result, event_data, None

    async def _audit_tool(
        self,
        spec: AgentRunSpec,
        hook_manager: HookManager | None,
        iteration: int | None,
        context: ToolRequestContext,
        decision: ToolPolicyDecision,
        **fields: Any,
    ) -> ToolAuditWriteResult | None:
        if not spec.policy_engine:
            return None
        record = spec.policy_engine.audit_record(context, decision, **fields)
        write_result = await spec.policy_engine.audit.write(record)
        if write_result is None:
            # Compatibility for custom sinks implementing the previous protocol.
            write_result = ToolAuditWriteResult(
                True, spec.policy_engine.redactor.redact(record)
            )
        if hook_manager:
            await hook_manager.dispatch(
                LifecycleEvent(
                    LifecycleEventType.TOOL_AUDIT,
                    payload={
                        "record": write_result.record,
                        "write_succeeded": write_result.succeeded,
                        "degraded": not write_result.succeeded,
                        "error": write_result.error,
                    },
                    phase="after",
                    run_id=spec.run_id,
                    session_key=spec.session_key,
                    iteration=iteration,
                    state=spec.event_metadata,
                )
            )
        return write_result

    @staticmethod
    def _mark_audit_degraded(
        event_data: dict[str, Any], write_result: ToolAuditWriteResult | None
    ) -> None:
        if write_result is not None and not write_result.succeeded:
            event_data["audit_status"] = "degraded"
            event_data["audit_error"] = write_result.error

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((perf_counter() - started) * 1000, 3)
