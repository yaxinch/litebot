"""Tool for paged retrieval of offloaded tool results."""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.session.artifacts import ToolArtifactStore


class GetToolResultTool(Tool):
    def __init__(self, store: ToolArtifactStore, default_page_size: int = 4096):
        self.store = store
        self.default_page_size = default_page_size
        self._session_key: ContextVar[str | None] = ContextVar(
            "tool_result_session_key", default=None,
        )

    @property
    def name(self) -> str:
        return "get_tool_result"

    @property
    def description(self) -> str:
        return "Read a page from a large tool result offloaded for the active session."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "default": self.default_page_size},
            },
            "required": ["artifact_id"],
        }

    def set_context(self, session_key: str) -> None:
        self._session_key.set(session_key)

    async def execute(self, artifact_id: str, offset: int = 0, limit: int | None = None) -> str:
        session_key = self._session_key.get()
        if not session_key:
            return "Error: no active session for artifact retrieval"
        try:
            page = self.store.get(
                session_key, artifact_id, offset, limit or self.default_page_size,
            )
            return json.dumps(page, ensure_ascii=False)
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            return f"Error: {exc}"
