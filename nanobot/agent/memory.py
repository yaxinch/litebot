"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import contextvars
import json
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.agent.episodic_memory import EpisodicMemorySource, EpisodicMemoryStore
from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entries": {
                        "type": "array",
                        "description": "Structured episodic events worth recalling in future conversations.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamp": {"type": "string", "description": "ISO-8601 event time."},
                                "category": {
                                    "type": "string",
                                    "enum": ["decision", "constraint", "preference", "project_state", "task_outcome", "fact"],
                                },
                                "content": {"type": "string"},
                                "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                            },
                            "required": ["timestamp", "category", "content", "importance"],
                        },
                    },
                    "history_entry": {
                        "type": "string",
                        "description": "Deprecated compatibility field. Prefer history_entries.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["memory_update"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None

_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)

_CONSOLIDATION_SOURCE: contextvars.ContextVar[EpisodicMemorySource | None] = (
    contextvars.ContextVar("nanobot_consolidation_source", default=None)
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class MemoryStore:
    """Long-term Markdown memory plus structured, retrievable episodic history."""

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.episodic = EpisodicMemoryStore(workspace)
        self._consecutive_failures = 0

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
        source: EpisodicMemorySource | None = None,
    ) -> bool:
        """Consolidate the provided message chunk into MEMORY.md + HISTORY.md."""
        if not messages:
            return True

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{self._format_messages(messages)}"""

        chat_messages = [
            {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation."},
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_memory"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(
                response.content
            ):
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Memory consolidation: LLM did not call save_memory "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_archive(messages, source)

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                return self._fail_or_raw_archive(messages, source)

            if "memory_update" not in args or not ({"history_entries", "history_entry"} & args.keys()):
                logger.warning("Memory consolidation: save_memory payload missing required fields")
                return self._fail_or_raw_archive(messages, source)

            update = args["memory_update"]

            if update is None:
                logger.warning("Memory consolidation: save_memory payload contains null required fields")
                return self._fail_or_raw_archive(messages, source)

            actual_source = source or EpisodicMemorySource()
            if "history_entries" in args:
                entries = args["history_entries"]
                if not isinstance(entries, list) or not entries:
                    logger.warning("Memory consolidation: history_entries must be a non-empty list")
                    return self._fail_or_raw_archive(messages, source)
                self.episodic.append(entries, source=actual_source)
            else:
                entry = args.get("history_entry")
                if entry is None:
                    logger.warning("Memory consolidation: history_entry is null")
                    return self._fail_or_raw_archive(messages, source)
                entry_text = _ensure_text(entry).strip()
                if not entry_text:
                    logger.warning("Memory consolidation: history_entry is empty after normalization")
                    return self._fail_or_raw_archive(messages, source)
                self.episodic.append(
                    [entry], source=actual_source, mirror=False,
                )
                # Preserve the historical Markdown representation for old providers/tests.
                self.append_history(entry_text)
            update = _ensure_text(update)
            if update != current_memory:
                self.write_long_term(update)

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(messages, source)

    def _fail_or_raw_archive(
        self, messages: list[dict], source: EpisodicMemorySource | None = None,
    ) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages, source)
        self._consecutive_failures = 0
        return True

    def _raw_archive(
        self, messages: list[dict], source: EpisodicMemorySource | None = None,
    ) -> None:
        """Fallback: dump raw messages to HISTORY.md without LLM summarization."""
        now = datetime.now(timezone.utc)
        raw = (
            f"[{now.strftime('%Y-%m-%d %H:%M')}] [RAW] {len(messages)} messages\n"
            f"{self._format_messages(messages)}"
        )
        self.episodic.append([{
            "timestamp": now.isoformat(),
            "category": "raw_archive",
            "content": raw,
            "importance": 2,
        }], source=source or _CONSOLIDATION_SOURCE.get() or EpisodicMemorySource(kind="raw_archive"), mirror=False)
        self.append_history(raw)
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
    ):
        self.store = MemoryStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive a selected message chunk into persistent memory."""
        return await self.store.consolidate(
            messages, self.provider, self.model, source=_CONSOLIDATION_SOURCE.get(),
        )

    async def _consolidate_with_source(
        self, messages: list[dict[str, object]], source: EpisodicMemorySource,
    ) -> bool:
        token = _CONSOLIDATION_SOURCE.set(source)
        try:
            return await self.consolidate_messages(messages)
        finally:
            _CONSOLIDATION_SOURCE.reset(token)

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(
        self, messages: list[dict[str, object]], *, session_key: str | None = None,
    ) -> bool:
        """Archive messages with guaranteed persistence (retries until raw-dump fallback)."""
        if not messages:
            return True
        source = EpisodicMemorySource(
            kind="session_archive", session_key=session_key,
            message_start=0, message_end=len(messages),
        )
        for _ in range(self.store._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            if await self._consolidate_with_source(messages, source):
                return True
        return True

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = budget // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < budget:
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                memory_source = EpisodicMemorySource(
                    kind="consolidation", session_key=session.key,
                    message_start=session.last_consolidated, message_end=end_idx,
                )
                if not await self._consolidate_with_source(chunk, memory_source):
                    return
                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
