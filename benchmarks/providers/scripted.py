from __future__ import annotations

from copy import deepcopy
from typing import Any

from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest


class ScriptedProvider(LLMProvider):
    """Deterministic offline provider returning a fixed response sequence."""

    def __init__(self, responses: list[dict[str, Any]], model: str = "scripted-v1"):
        super().__init__()
        self.responses = deepcopy(responses)
        self.model = model
        self.index = 0
        self.generation = GenerationSettings(temperature=0, max_tokens=512)

    async def chat(self, messages, tools=None, model=None, **kwargs) -> LLMResponse:
        if self.index >= len(self.responses):
            raise RuntimeError("ScriptedProvider response sequence exhausted")
        raw = self.responses[self.index]
        self.index += 1
        calls = [ToolCallRequest(id=item["id"], name=item["name"], arguments=item.get("arguments", {})) for item in raw.get("tool_calls", [])]
        return LLMResponse(
            content=raw.get("content"), tool_calls=calls,
            finish_reason=raw.get("finish_reason", "stop"),
            usage=raw.get("usage", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        )

    def get_default_model(self) -> str:
        return self.model

    def estimate_prompt_tokens(self, *args: Any, **kwargs: Any):
        return 100, "scripted"

