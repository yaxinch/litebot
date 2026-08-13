from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool


class EchoTool(Tool):
    name = "echo"
    description = "Return the supplied value unchanged."
    parameters = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}

    async def execute(self, value: str, **kwargs: Any) -> str:
        return value


class LookupTool(Tool):
    name = "lookup"
    description = "Look up a deterministic fixture key."
    parameters = {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}

    async def execute(self, key: str, **kwargs: Any) -> str:
        return {"alpha": "A-17", "beta": "B-29", "gamma": "G-41"}.get(key, f"unknown:{key}")


class ErrorTool(Tool):
    name = "error_tool"
    description = "Return a deterministic error."
    parameters = {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}

    async def execute(self, reason: str, **kwargs: Any) -> str:
        return f"Error: {reason}"


class ExplodingTool(Tool):
    name = "exploding_tool"
    description = "Raise a deterministic exception."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        raise RuntimeError("deterministic boom")

