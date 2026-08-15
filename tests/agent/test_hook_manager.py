"""Lifecycle HookManager contracts."""

import asyncio

import pytest

from nanobot.agent.hook import (
    HookAction,
    HookDecision,
    HookManager,
    LifecycleEvent,
    LifecycleEventType,
)


@pytest.mark.asyncio
async def test_handlers_are_priority_ordered_and_apply_patches():
    manager = HookManager()
    calls: list[str] = []

    def low(event):
        calls.append("low")
        return HookDecision.modify({"value": event.payload["value"] + 1})

    async def high(event):
        calls.append("high")
        return HookDecision.modify({"value": 2})

    manager.register(LifecycleEventType.PRE_LLM_CALL, low, priority=0)
    manager.register(LifecycleEventType.PRE_LLM_CALL, high, priority=10)
    event = await manager.dispatch(LifecycleEvent(LifecycleEventType.PRE_LLM_CALL, {"value": 0}))

    assert calls == ["high", "low"]
    assert event.payload["value"] == 3


@pytest.mark.asyncio
async def test_deny_short_circuits_and_unregister_is_idempotent():
    manager = HookManager()
    calls: list[str] = []
    registration = manager.register(
        LifecycleEventType.PRE_TOOL_USE,
        lambda event: (calls.append("deny"), HookDecision.deny("blocked"))[1],
    )
    manager.register(LifecycleEventType.PRE_TOOL_USE, lambda event: calls.append("later"))
    event = await manager.dispatch(LifecycleEvent(LifecycleEventType.PRE_TOOL_USE))

    assert event.decision and event.decision.action is HookAction.DENY
    assert calls == ["deny"]
    assert manager.unregister(registration)
    assert not manager.unregister(registration)


@pytest.mark.asyncio
async def test_failures_are_isolated_but_cancellation_propagates():
    manager = HookManager()
    calls: list[str] = []

    def broken(event):
        raise RuntimeError("boom")

    manager.register(LifecycleEventType.POST_TOOL_USE, broken)
    manager.register(LifecycleEventType.POST_TOOL_USE, lambda event: calls.append("ok"))
    event = await manager.dispatch(LifecycleEvent(LifecycleEventType.POST_TOOL_USE))
    assert calls == ["ok"]
    assert event.failures[0].stage == "handler"

    async def cancelled(event):
        raise asyncio.CancelledError()

    manager.register(LifecycleEventType.SESSION_END, cancelled)
    with pytest.raises(asyncio.CancelledError):
        await manager.dispatch(LifecycleEvent(LifecycleEventType.SESSION_END))
