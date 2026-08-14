"""Budget-aware prompt management independent from the product agent loop."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nanobot.agent.context_summary import SUMMARY_HEADING, ContextSummarizer
from nanobot.config.schema import ContextManagementConfig
from nanobot.session.artifacts import ToolArtifactStore
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.helpers import estimate_prompt_tokens_chain


class ContextOverflowError(RuntimeError):
    """Raised when protected prompt content cannot fit the configured context."""


@dataclass(frozen=True, slots=True)
class ContextManagementPolicy:
    recent_turns: int = 8
    soft_threshold: float = 0.80
    hard_threshold: float = 0.92
    compaction_target: float = 0.68
    safety_margin_tokens: int = 1024
    output_reserve_tokens: int = 4096
    tool_offload_threshold_bytes: int = 8192
    tool_summary_max_chars: int = 1000
    artifact_page_size: int = 4096
    artifact_max_reads_per_session: int = 3
    artifact_max_searches_per_session: int = 5
    artifact_max_returned_chars_per_session: int = 12_288
    artifact_max_sequential_reads: int = 3
    artifact_ttl_days: int = 30
    artifact_gc_interval_seconds: int = 86400
    max_compaction_rounds: int = 5

    @classmethod
    def from_config(cls, config: ContextManagementConfig | None, default_output_reserve: int) -> "ContextManagementPolicy":
        cfg = config or ContextManagementConfig()
        if (
            not isinstance(default_output_reserve, int)
            or isinstance(default_output_reserve, bool)
            or default_output_reserve <= 0
        ):
            default_output_reserve = 4096
        values: dict[str, Any] = {
            "recent_turns": cfg.recent_turns,
            "soft_threshold": cfg.soft_threshold,
            "hard_threshold": cfg.hard_threshold,
            "compaction_target": cfg.compaction_target,
            "safety_margin_tokens": cfg.safety_margin_tokens,
            "output_reserve_tokens": cfg.output_reserve_tokens or default_output_reserve,
            "tool_offload_threshold_bytes": cfg.tool_offload_threshold_bytes,
            "tool_summary_max_chars": cfg.tool_summary_max_chars,
            "artifact_page_size": cfg.artifact_page_size,
            "artifact_max_reads_per_session": cfg.artifact_max_reads_per_session,
            "artifact_max_searches_per_session": cfg.artifact_max_searches_per_session,
            "artifact_max_returned_chars_per_session": cfg.artifact_max_returned_chars_per_session,
            "artifact_max_sequential_reads": cfg.artifact_max_sequential_reads,
            "artifact_ttl_days": cfg.artifact_ttl_days,
            "artifact_gc_interval_seconds": cfg.artifact_gc_interval_seconds,
            "max_compaction_rounds": cfg.max_compaction_rounds,
        }
        casts = {
            "recent_turns": int, "soft_threshold": float, "hard_threshold": float,
            "compaction_target": float, "safety_margin_tokens": int,
            "output_reserve_tokens": int, "tool_offload_threshold_bytes": int,
            "tool_summary_max_chars": int, "artifact_page_size": int,
            "artifact_max_reads_per_session": int,
            "artifact_max_searches_per_session": int,
            "artifact_max_returned_chars_per_session": int,
            "artifact_max_sequential_reads": int,
            "artifact_ttl_days": int, "artifact_gc_interval_seconds": int,
            "max_compaction_rounds": int,
        }
        for key, cast in casts.items():
            raw = os.environ.get(f"NANOBOT_CONTEXT_{key.upper()}")
            if raw is not None:
                values[key] = cast(raw)
        policy = cls(**values)
        if not 0 < policy.compaction_target < policy.soft_threshold < policy.hard_threshold <= 1:
            raise ValueError("context thresholds must satisfy 0 < target < soft < hard <= 1")
        if any(getattr(policy, key) <= 0 for key in casts if "threshold" not in key):
            raise ValueError("context count and size settings must be positive")
        return policy


@dataclass(slots=True)
class PreparedContext:
    messages: list[dict[str, Any]]
    estimated_tokens: int
    token_source: str
    compacted_turns: int = 0
    hard_truncated_turns: int = 0
    offloaded_artifacts: int = 0
    actions: list[str] = field(default_factory=list)


class ContextManager:
    def __init__(
        self, provider: Any, model: str, context_window_tokens: int,
        policy: ContextManagementPolicy, artifacts: ToolArtifactStore,
        sessions: SessionManager | None = None,
    ):
        self.provider = provider
        self.model = model
        self.context_window_tokens = context_window_tokens
        self.policy = policy
        self.artifacts = artifacts
        self.sessions = sessions
        self.summarizer = ContextSummarizer(provider, model)
        self.last_prepared: PreparedContext | None = None
        self.telemetry: list[dict[str, Any]] = []
        self.total_offloaded_artifacts = 0

    @property
    def effective_budget(self) -> int:
        return max(1, self.context_window_tokens - self.policy.output_reserve_tokens - self.policy.safety_margin_tokens)

    def _estimate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[int, str]:
        return estimate_prompt_tokens_chain(self.provider, self.model, messages, tools)

    def offload_tool_results(self, messages: list[dict[str, Any]], session_key: str) -> int:
        count = 0
        for message in messages:
            if message.get("role") != "tool" or str(message.get("content", "")).startswith("[Tool Result Offloaded]"):
                continue
            payload, _ = self.artifacts.serialize(message.get("content"))
            if len(payload) <= self.policy.tool_offload_threshold_bytes:
                continue
            try:
                artifact = self.artifacts.put(
                    session_key, str(message.get("name", "tool")),
                    str(message.get("tool_call_id", "")), message.get("content"),
                )
                text = payload.decode("utf-8", errors="replace")
                preview = text[:self.policy.tool_summary_max_chars]
                message["content"] = (
                    "[Tool Result Offloaded]\n"
                    f"artifact_id: {artifact.artifact_id}\n"
                    f"content_type: {artifact.content_type}\n"
                    f"original_size_bytes: {artifact.size_bytes}\n"
                    f"line_count: {artifact.line_count}\n"
                    f"tool: {artifact.tool_name}\n"
                    f"sha256: {artifact.sha256}\nsummary: {preview}\n"
                    "retrieval: Unknown location -> search_tool_result(artifact_id, query). "
                    "For an unknown standalone identifier/marker, search for a likely literal such as "
                    "RESULT or for the newline character to inspect bounded line boundaries. "
                    "Known location or more local context -> "
                    f'get_tool_result(artifact_id="{artifact.artifact_id}", offset=<known_offset>, limit<={self.policy.artifact_page_size}).'
                )
                count += 1
            except Exception as exc:
                preview = payload.decode("utf-8", errors="replace")[:self.policy.tool_summary_max_chars]
                message["content"] = f"[Tool Result Offload Failed: {type(exc).__name__}]\n{preview}"
        self.total_offloaded_artifacts += count
        return count

    @staticmethod
    def _summary_index(messages: list[dict[str, Any]]) -> int | None:
        for idx, message in enumerate(messages):
            if message.get("role") == "system" and str(message.get("content", "")).startswith(SUMMARY_HEADING):
                return idx
        return None

    @staticmethod
    def _turn_ranges(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
        starts = [idx for idx, message in enumerate(messages) if message.get("role") == "user"]
        return [(start, starts[i + 1] if i + 1 < len(starts) else len(messages)) for i, start in enumerate(starts)]

    @staticmethod
    def validate_tool_structure(messages: list[dict[str, Any]]) -> bool:
        pending: set[str] = set()
        for message in messages:
            role = message.get("role")
            if role == "assistant":
                if pending:
                    return False
                pending = {str(tc.get("id")) for tc in message.get("tool_calls") or [] if tc.get("id")}
            elif role == "tool":
                tool_id = str(message.get("tool_call_id", ""))
                if tool_id not in pending:
                    return False
                pending.remove(tool_id)
            elif role in {"user", "system"} and pending:
                return False
        return not pending

    async def prepare(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        session: Session | None = None, session_key: str | None = None,
    ) -> PreparedContext:
        work = messages
        offloaded = self.offload_tool_results(work, session_key) if session_key else 0
        estimated, source = self._estimate(work, tools)
        prepared = PreparedContext(work, estimated, source, offloaded_artifacts=offloaded)
        soft_limit = int(self.effective_budget * self.policy.soft_threshold)
        target = int(self.effective_budget * self.policy.compaction_target)
        hard_limit = int(self.effective_budget * self.policy.hard_threshold)

        if estimated > soft_limit and session is not None:
            for _ in range(self.policy.max_compaction_rounds):
                ranges = session.context_turn_ranges()
                eligible = ranges[:-self.policy.recent_turns] if len(ranges) > self.policy.recent_turns else []
                if not eligible or estimated <= target:
                    break
                start, end = eligible[0][0], eligible[-1][1]
                chunk = Session._to_llm_messages(session.messages[start:end])
                summary = await self.summarizer.summarize(session.context_summary, chunk)
                if not summary:
                    prepared.actions.append("summary_failed")
                    break
                old_through = session.context_summary_through
                visible_start = max(old_through, session.last_consolidated)
                session.context_summary = summary
                session.context_summary_through = end
                session.context_summary_updated_at = datetime.now()
                if self.sessions:
                    self.sessions.save(session)
                remove_count = max(0, end - visible_start)
                summary_idx = self._summary_index(work)
                history_start = (summary_idx + 1) if summary_idx is not None else 1
                del work[history_start:history_start + remove_count]
                summary_message = {"role": "system", "content": f"{SUMMARY_HEADING}\n{summary}"}
                if summary_idx is None:
                    work.insert(1, summary_message)
                else:
                    work[summary_idx] = summary_message
                prepared.compacted_turns += len(eligible)
                prepared.actions.append("compacted")
                estimated, source = self._estimate(work, tools)
                break

        while estimated > hard_limit:
            ranges = self._turn_ranges(work)
            if len(ranges) <= 1:
                raise ContextOverflowError(
                    f"protected context requires {estimated} tokens, hard budget is {hard_limit}"
                )
            start, end = ranges[0]
            del work[start:end]
            prepared.hard_truncated_turns += 1
            prepared.actions.append("hard_truncated")
            estimated, source = self._estimate(work, tools)

        if not self.validate_tool_structure(work):
            raise ContextOverflowError("context management produced an invalid tool-call sequence")
        prepared.estimated_tokens = estimated
        prepared.token_source = source
        self.last_prepared = prepared
        self.telemetry.append({
            "estimated_tokens": estimated,
            "token_source": source,
            "compacted_turns": prepared.compacted_turns,
            "hard_truncated_turns": prepared.hard_truncated_turns,
            "offloaded_artifacts": prepared.offloaded_artifacts,
            "actions": list(prepared.actions),
        })
        return prepared
