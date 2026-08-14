"""Tool for paged retrieval of offloaded tool results."""

from __future__ import annotations

import json
import re
import unicodedata
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.session.artifacts import ToolArtifactStore

_DISTINCTIVE_IDENTIFIER = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){2,}\b")


@dataclass(slots=True)
class _RetrievalState:
    reads: int = 0
    searches: int = 0
    returned_chars: int = 0
    last_end: int | None = None
    sequential_reads: int = 0
    search_receipts: dict[tuple[tuple[str, ...], int, int], dict[str, Any]] = field(default_factory=dict)
    retrieved_ranges: list[tuple[int, int, str]] = field(default_factory=list)
    answer_ready: bool = False
    answer_candidates: list[str] = field(default_factory=list)
    answer_offsets: list[int] = field(default_factory=list)
    answer_local_get_used: bool = False


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

    def read_receipt(
        self, session_key: str, artifact_id: str, offset: int, limit: int,
    ) -> dict[str, Any] | None:
        state = self._states.get((session_key, artifact_id))
        if not state:
            return None
        requested_end = offset + limit
        overlaps = [
            (start, end, source) for start, end, source in state.retrieved_ranges
            if offset < end and start < requested_end
        ]
        # A local get after search is an intentional supported workflow. Only prior
        # page reads suppress duplicate/overlapping page reads.
        if not any(source == "get" for _, _, source in overlaps):
            return None
        exact = any(start == offset and end == requested_end for start, end, _ in overlaps)
        return {
            "status": "cached_receipt" if exact else "overlap_already_retrieved",
            "artifact_id": artifact_id,
            "requested_range": [offset, requested_end],
            "overlapping_ranges": [
                {"start": start, "end": end, "source": source}
                for start, end, source in overlaps
            ],
            "hint": (
                "This range was already returned. Use the retained retrieval result or search a "
                "distinctive literal instead of rereading overlapping content."
            ),
        }

    def convergence_get_receipt(
        self, session_key: str, artifact_id: str, offset: int, limit: int,
    ) -> dict[str, Any] | None:
        state = self._states.get((session_key, artifact_id))
        if not state or not state.answer_ready:
            return None
        covers_match = any(offset <= match_offset < offset + limit for match_offset in state.answer_offsets)
        if covers_match and not state.answer_local_get_used:
            return None
        return {
            "status": "answer_ready_receipt",
            "artifact_id": artifact_id,
            "answer_ready": True,
            "answer_candidates": state.answer_candidates,
            "match_offsets": state.answer_offsets,
            "hint": (
                "The target identifier is already available. Only one local get covering a known "
                "match offset is permitted; answer from the retained candidate instead."
            ),
        }

    def record(self, session_key: str, artifact_id: str, offset: int, end: int) -> dict[str, int]:
        state = self._states.setdefault((session_key, artifact_id), _RetrievalState())
        state.sequential_reads = state.sequential_reads + 1 if state.last_end == offset else 1
        state.reads += 1
        state.returned_chars += end - offset
        state.last_end = end
        state.retrieved_ranges.append((offset, end, "get"))
        if state.answer_ready:
            state.answer_local_get_used = True
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

    def search_receipt(
        self, session_key: str, artifact_id: str, queries: tuple[str, ...],
        max_matches: int, context_chars: int,
    ) -> dict[str, Any] | None:
        state = self._states.get((session_key, artifact_id))
        if not state:
            return None
        if state.answer_ready:
            return {
                "status": "answer_ready_receipt",
                "artifact_id": artifact_id,
                "queries": list(queries),
                "answer_ready": True,
                "answer_candidates": state.answer_candidates,
                "match_offsets": state.answer_offsets,
                "hint": "A high-confidence identifier was already found; answer without another search.",
            }
        receipt = state.search_receipts.get((queries, max_matches, context_chars))
        if receipt is None:
            return None
        return {
            "status": "cached_receipt",
            "artifact_id": artifact_id,
            "queries": list(queries),
            **receipt,
            "hint": (
                "This exact search was already completed. Reuse the retained result; do not "
                "repeat the same artifact/query."
            ),
        }

    def record_search(
        self, session_key: str, artifact_id: str, returned_chars: int, matched: bool,
        queries: tuple[str, ...] | None = None, match_offsets: list[int] | None = None,
        max_matches: int = 5, context_chars: int = 500,
        snippet_ranges: list[tuple[int, int]] | None = None,
        answer_candidates: list[str] | None = None,
        answer_offsets: list[int] | None = None,
    ) -> dict[str, int]:
        state = self._states.setdefault((session_key, artifact_id), _RetrievalState())
        state.searches += 1
        state.returned_chars += returned_chars
        if queries is not None:
            state.search_receipts[(queries, max_matches, context_chars)] = {
                "original_status": "matches" if matched else "no_match",
                "match_offsets": list(match_offsets or []),
            }
        for start, end in snippet_ranges or []:
            state.retrieved_ranges.append((start, end, "search"))
        if matched:
            state.reads = 0
            state.last_end = None
            state.sequential_reads = 0
        if answer_candidates:
            state.answer_ready = True
            state.answer_candidates = list(answer_candidates)
            state.answer_offsets = list(answer_offsets or match_offsets or [])
            state.answer_local_get_used = False
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
        return (
            "Read one bounded local page at a known character offset from an offloaded tool result. "
            "When the location is unknown, first call search_tool_result once with several reasonable "
            "queries. Offset 0 is valid when it is already the known target location; do not use it for "
            "blind scanning. After answer_ready, at most one get covering a returned match offset is useful."
        )

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
            if receipt := self.guard.convergence_get_receipt(
                session_key, artifact_id, offset, requested,
            ):
                return json.dumps(receipt, ensure_ascii=False)
            if receipt := self.guard.read_receipt(
                session_key, artifact_id, offset, requested,
            ):
                return json.dumps(receipt, ensure_ascii=False)
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
    def __init__(
        self, store: ToolArtifactStore, guard: ArtifactRetrievalGuard,
        total_snippet_chars: int = 2_000,
    ):
        self.store = store
        self.guard = guard
        self.total_snippet_chars = total_snippet_chars
        self._session_key: ContextVar[str | None] = ContextVar(
            "search_tool_result_session_key", default=None,
        )

    @property
    def name(self) -> str:
        return "search_tool_result"

    @property
    def description(self) -> str:
        return (
            "Locate unknown content in a session-owned UTF-8 artifact without paging through it. "
            "Provide query or up to five queries (for example MARKER, RESULT, LARGE-RESULT) in one "
            "server-side search. If answer_ready is true, answer directly. Only when extra context is "
            "essential, call get_tool_result once near a returned offset."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "query": {"type": "string", "minLength": 1, "maxLength": 256},
                "queries": {
                    "type": "array", "minItems": 1, "maxItems": 5,
                    "items": {"type": "string", "minLength": 1, "maxLength": 256},
                },
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 5, "default": 5},
                "context_chars": {"type": "integer", "minimum": 0, "maximum": 500, "default": 500},
            },
            "required": ["artifact_id"],
            "oneOf": [{"required": ["query"]}, {"required": ["queries"]}],
        }

    def set_context(self, session_key: str) -> None:
        self._session_key.set(session_key)

    async def execute(
        self, artifact_id: str, query: str | None = None,
        queries: list[str] | None = None, max_matches: int = 5,
        context_chars: int = 500,
    ) -> str:
        session_key = self._session_key.get()
        if not session_key:
            return "Error: no active session for artifact retrieval"
        try:
            self.store.validate_artifact_id(artifact_id)
            if (query is None) == (queries is None):
                return "Error: provide exactly one of query or queries"
            requested_queries = [query] if query is not None else list(queries or [])
            if len(requested_queries) > 5:
                return "Error: queries must contain at most 5 items"
            valid_queries: list[str] = []
            rejected_queries: list[dict[str, str]] = []
            for item in requested_queries:
                reason = self._query_rejection_reason(item)
                if reason:
                    rejected_queries.append({"query": item, "reason": reason})
                elif item not in valid_queries:
                    valid_queries.append(item)
            if not valid_queries:
                return json.dumps({
                    "status": "query_too_broad",
                    "artifact_id": artifact_id,
                    "matches": [],
                    "rejected_queries": rejected_queries,
                    "hint": "Use one or more specific literal words or identifiers without newlines.",
                }, ensure_ascii=False)
            query_key = tuple(valid_queries)
            if receipt := self.guard.search_receipt(
                session_key, artifact_id, query_key, max_matches, context_chars,
            ):
                if rejected_queries:
                    receipt["rejected_queries"] = rejected_queries
                return json.dumps(receipt, ensure_ascii=False)
            remaining_chars, reason = self.guard.allow_search(session_key, artifact_id)
            if reason:
                return (
                    f"Error: artifact retrieval guard blocked search_tool_result ({reason}). "
                    "Use a prior match offset with get_tool_result or answer from already returned snippets."
                )
            if remaining_chars < max(len(item) for item in valid_queries):
                return (
                    "Error: artifact retrieval guard blocked search_tool_result "
                    "(result would exceed maximum returned artifact characters). "
                    "Answer from prior results or use a shorter specific query."
                )
            result = self.store.search_many(
                session_key, artifact_id, valid_queries, max_matches, context_chars,
                min(self.total_snippet_chars, remaining_chars),
            )
            if query is not None:
                result["query"] = query
            result["rejected_queries"] = rejected_queries
            returned_chars = sum(len(match["snippet"]) for match in result["matches"])
            if returned_chars > remaining_chars:
                return (
                    "Error: artifact retrieval guard blocked search_tool_result "
                    "(result would exceed maximum returned artifact characters). "
                    "Retry with smaller context_chars or max_matches."
                )
            answer_offsets: list[int] = []
            if result["matches"]:
                candidates: set[str] = set()
                for match in result["matches"]:
                    snippet = match["snippet"]
                    for candidate_match in _DISTINCTIVE_IDENTIFIER.finditer(snippet):
                        truncated_left = (
                            candidate_match.start() == 0 and int(match["snippet_start"]) > 0
                        )
                        truncated_right = (
                            candidate_match.end() == len(snippet)
                            and int(match["snippet_end"]) < int(result["total_chars"])
                        )
                        if not truncated_left and not truncated_right:
                            candidates.add(candidate_match.group(0))
                            answer_offsets.append(
                                int(match["snippet_start"]) + candidate_match.start()
                            )
                sorted_candidates = sorted(candidates)
                result["answer_ready"] = bool(sorted_candidates)
                result["answer_candidates"] = sorted_candidates[:5]
                result["hint"] = (
                    "A distinctive identifier candidate is visible. Answer now if it satisfies "
                    "the request; otherwise use at most one local get_tool_result call covering "
                    "its match offset. Do not issue another broad search for this artifact."
                    if sorted_candidates else
                    "These matches do not expose a distinctive identifier candidate. For an "
                    "unknown exact marker or identifier, search for the literal RESULT next; "
                    "do not guess more generic queries or page through the artifact."
                    )
            else:
                result["answer_ready"] = False
                result["answer_candidates"] = []
            result["retrieval_guard"] = self.guard.record_search(
                session_key, artifact_id, returned_chars, bool(result["matches"]),
                queries=query_key,
                match_offsets=[int(match["offset"]) for match in result["matches"]],
                max_matches=max_matches,
                context_chars=context_chars,
                snippet_ranges=[
                    (int(match["snippet_start"]), int(match["snippet_end"]))
                    for match in result["matches"]
                ],
                answer_candidates=result.get("answer_candidates"),
                answer_offsets=sorted(set(
                    answer_offsets
                    + [int(match["offset"]) for match in result["matches"]]
                )),
            )
            if not result["matches"]:
                result["hint"] = (
                    "Try several distinctive literals expected in the target in one queries call, "
                    "such as MARKER, RESULT, and a domain-specific identifier prefix."
                )
            return json.dumps(result, ensure_ascii=False)
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            return f"Error: {exc}"

    @staticmethod
    def _query_rejection_reason(query: Any) -> str | None:
        if not isinstance(query, str):
            return "not_a_string"
        if not query or not query.strip():
            return "empty_or_whitespace"
        if "\n" in query or "\r" in query:
            return "newline_not_allowed"
        if len(query) > 256:
            return "too_long"
        stripped = query.strip()
        if len(stripped) == 1 and stripped.isascii():
            return "low_information_ascii_single_character"
        if all(unicodedata.category(char)[0] in {"P", "S", "Z", "C"} for char in stripped):
            return "punctuation_or_symbol_only"
        return None
