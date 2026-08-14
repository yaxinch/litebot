"""Tool for paged retrieval of offloaded tool results."""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.session.artifacts import ToolArtifactStore


@dataclass(slots=True)
class _RetrievalState:
    reads: int = 0
    searches: int = 0
    returned_chars: int = 0
    last_end: int | None = None
    sequential_reads: int = 0


class ArtifactRetrievalGuard:
    """Small in-memory guard against blind artifact pagination within a session."""

    def __init__(
        self, max_reads: int = 3, max_searches: int = 5,
        max_returned_chars: int = 12_288,
        max_sequential_reads: int = 3,
    ):
        self.max_reads = max_reads
        self.max_searches = max_searches
        self.max_returned_chars = max_returned_chars
        self.max_sequential_reads = max_sequential_reads
        self._states: dict[tuple[str, str], _RetrievalState] = {}

    def allow(self, session_key: str, artifact_id: str, offset: int, limit: int) -> tuple[int, str | None]:
        state = self._states.setdefault((session_key, artifact_id), _RetrievalState())
        sequential = state.last_end is not None and offset == state.last_end
        next_sequential = state.sequential_reads + 1 if sequential else 1
        if state.reads >= self.max_reads:
            return 0, "maximum reads reached"
        if state.returned_chars >= self.max_returned_chars:
            return 0, "maximum returned artifact characters reached"
        if sequential and next_sequential > self.max_sequential_reads:
            return 0, "sequential offset scan detected"
        return min(limit, self.max_returned_chars - state.returned_chars), None

    def record(self, session_key: str, artifact_id: str, offset: int, end: int) -> dict[str, int]:
        state = self._states.setdefault((session_key, artifact_id), _RetrievalState())
        state.sequential_reads = state.sequential_reads + 1 if state.last_end == offset else 1
        state.reads += 1
        state.returned_chars += end - offset
        state.last_end = end
        return {
            "reads_remaining": max(0, self.max_reads - state.reads),
            "chars_remaining": max(0, self.max_returned_chars - state.returned_chars),
        }

    def allow_search(self, session_key: str, artifact_id: str) -> tuple[int, str | None]:
        state = self._states.setdefault((session_key, artifact_id), _RetrievalState())
        if state.searches >= self.max_searches:
            return 0, "maximum searches reached"
        if state.returned_chars >= self.max_returned_chars:
            return 0, "maximum returned artifact characters reached"
        return self.max_returned_chars - state.returned_chars, None

    def record_search(
        self, session_key: str, artifact_id: str, returned_chars: int, matched: bool,
    ) -> dict[str, int]:
        state = self._states.setdefault((session_key, artifact_id), _RetrievalState())
        state.searches += 1
        state.returned_chars += returned_chars
        if matched:
            state.reads = 0
            state.last_end = None
            state.sequential_reads = 0
        return {
            "searches_remaining": max(0, self.max_searches - state.searches),
            "reads_remaining": max(0, self.max_reads - state.reads),
            "chars_remaining": max(0, self.max_returned_chars - state.returned_chars),
        }


class GetToolResultTool(Tool):
    def __init__(
        self, store: ToolArtifactStore, default_page_size: int = 4096,
        guard: ArtifactRetrievalGuard | None = None,
    ):
        self.store = store
        self.default_page_size = default_page_size
        self.guard = guard or ArtifactRetrievalGuard()
        self._session_key: ContextVar[str | None] = ContextVar(
            "tool_result_session_key", default=None,
        )

    @property
    def name(self) -> str:
        return "get_tool_result"

    @property
    def description(self) -> str:
        return "Read a local page at a known character offset from an offloaded tool result. Use search_tool_result first when the location is unknown."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": self.default_page_size, "default": self.default_page_size},
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
            requested = limit or self.default_page_size
            requested = min(requested, self.default_page_size)
            allowed, reason = self.guard.allow(session_key, artifact_id, offset, requested)
            if reason:
                return (
                    f"Error: artifact retrieval guard blocked get_tool_result ({reason}). "
                    "Use search_tool_result to locate the target, then get_tool_result near the matched offset."
                )
            page = self.store.get(
                session_key, artifact_id, offset, allowed,
            )
            page["retrieval_guard"] = self.guard.record(
                session_key, artifact_id, page["offset"], page["end"],
            )
            return json.dumps(page, ensure_ascii=False)
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            return f"Error: {exc}"


class SearchToolResultTool(Tool):
    def __init__(self, store: ToolArtifactStore, guard: ArtifactRetrievalGuard):
        self.store = store
        self.guard = guard
        self._session_key: ContextVar[str | None] = ContextVar(
            "search_tool_result_session_key", default=None,
        )

    @property
    def name(self) -> str:
        return "search_tool_result"

    @property
    def description(self) -> str:
        return "Search a session-owned UTF-8 text tool artifact and return bounded match snippets and character offsets."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "query": {"type": "string", "minLength": 1, "maxLength": 256},
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 5, "default": 5},
                "context_chars": {"type": "integer", "minimum": 0, "maximum": 500, "default": 500},
            },
            "required": ["artifact_id", "query"],
        }

    def set_context(self, session_key: str) -> None:
        self._session_key.set(session_key)

    async def execute(
        self, artifact_id: str, query: str, max_matches: int = 5,
        context_chars: int = 500,
    ) -> str:
        session_key = self._session_key.get()
        if not session_key:
            return "Error: no active session for artifact retrieval"
        try:
            remaining_chars, reason = self.guard.allow_search(session_key, artifact_id)
            if reason:
                return (
                    f"Error: artifact retrieval guard blocked search_tool_result ({reason}). "
                    "Use a prior match offset with get_tool_result or answer from already returned snippets."
                )
            result = self.store.search(
                session_key, artifact_id, query, max_matches, context_chars,
            )
            returned_chars = sum(len(match["snippet"]) for match in result["matches"])
            if returned_chars > remaining_chars:
                return (
                    "Error: artifact retrieval guard blocked search_tool_result "
                    "(result would exceed maximum returned artifact characters). "
                    "Retry with smaller context_chars or max_matches."
                )
            result["retrieval_guard"] = self.guard.record_search(
                session_key, artifact_id, returned_chars, bool(result["matches"]),
            )
            if not result["matches"]:
                result["hint"] = (
                    "Try a distinctive literal expected in the target. For an unknown standalone "
                    "identifier or marker, query RESULT or the newline character to inspect line boundaries."
                )
            return json.dumps(result, ensure_ascii=False)
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            return f"Error: {exc}"
