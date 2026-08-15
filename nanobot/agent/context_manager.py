"""Budget-aware prompt management independent from the product agent loop."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nanobot.agent.context_summary import SUMMARY_HEADING, ContextSummarizer
from nanobot.agent.episodic_memory import RetrievalResult
from nanobot.agent.hook import HookAction, HookManager, LifecycleEvent, LifecycleEventType
from nanobot.config.schema import ContextManagementConfig
from nanobot.session.artifacts import ToolArtifactStore
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.helpers import estimate_prompt_tokens_chain

_RETRIEVAL_TOOLS = {"get_tool_result", "search_tool_result"}
_RETRIEVAL_RECEIPT = "[Artifact Retrieval Compacted]"


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
    tool_summary_max_chars: int = 1024
    artifact_page_size: int = 4096
    artifact_search_total_snippet_chars: int = 2000
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
            "artifact_search_total_snippet_chars": cfg.artifact_search_total_snippet_chars,
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
            "artifact_search_total_snippet_chars": int,
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
    compacted_retrieval_results: int = 0
    episodic_entries: int = 0
    episodic_injected_chars: int = 0
    episodic_candidates: int = 0
    episodic_duplicates_suppressed: int = 0
    actions: list[str] = field(default_factory=list)


class ContextManager:
    def __init__(
        self, provider: Any, model: str, context_window_tokens: int,
        policy: ContextManagementPolicy, artifacts: ToolArtifactStore,
        sessions: SessionManager | None = None,
        hook_manager: HookManager | None = None,
    ):
        self.provider = provider
        self.model = model
        self.context_window_tokens = context_window_tokens
        self.policy = policy
        self.artifacts = artifacts
        self.sessions = sessions
        self.hook_manager = hook_manager
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
                    "retrieval: Unknown location -> call search_tool_result once with multiple likely "
                    "literals, for example queries=[\"MARKER\", \"RESULT\", \"LARGE-RESULT\"]. "
                    "When search returns answer_ready=true, answer directly. "
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
    def _retrieval_receipt(
        message: dict[str, Any], *, retain_excerpt: bool = False,
    ) -> str:
        content = message.get("content")
        try:
            result = json.loads(content) if isinstance(content, str) else content
        except (TypeError, json.JSONDecodeError):
            result = None
        if not isinstance(result, dict):
            return f"{_RETRIEVAL_RECEIPT}\ntool: {message.get('name', 'retrieval')}\nstatus: unavailable"

        lines = [
            _RETRIEVAL_RECEIPT,
            f"tool: {message.get('name', 'retrieval')}",
            f"artifact_id: {result.get('artifact_id', 'unknown')}",
            f"status: {result.get('status', 'ok')}",
        ]
        if "query" in result:
            lines.append(f"query: {json.dumps(result['query'], ensure_ascii=False)}")
        elif "queries" in result:
            lines.append(f"queries: {json.dumps(result['queries'], ensure_ascii=False)}")
        if "offset" in result:
            lines.append(f"range: [{result.get('offset')}, {result.get('end')})")
        offsets = [
            match.get("offset") for match in result.get("matches", [])
            if isinstance(match, dict) and match.get("offset") is not None
        ]
        if offsets:
            lines.append(f"match_offsets: {offsets}")
        if result.get("answer_ready"):
            lines.append("answer_ready: true")
        if result.get("answer_candidates"):
            lines.append(
                f"answer_candidates: {json.dumps(result['answer_candidates'], ensure_ascii=False)}"
            )
        if retain_excerpt:
            excerpt = ""
            matches = [match for match in result.get("matches", []) if isinstance(match, dict)]
            if matches and isinstance(matches[0].get("snippet"), str):
                match = matches[0]
                snippet = match["snippet"]
                relative = max(0, int(match.get("offset", 0)) - int(match.get("snippet_start", 0)))
                start = max(0, relative - 160)
                excerpt = snippet[start:start + 320]
            elif isinstance(result.get("content"), str):
                content_text = result["content"]
                excerpt = (
                    content_text if len(content_text) <= 320
                    else content_text[:160] + " … " + content_text[-160:]
                )
            if excerpt:
                lines.append(f"retained_excerpt: {json.dumps(excerpt, ensure_ascii=False)}")
        return "\n".join(lines)

    @staticmethod
    def compact_transient_retrieval_results(messages: list[dict[str, Any]]) -> int:
        """Compact old retrieval responses while retaining the latest useful match."""
        candidates: list[tuple[int, bool]] = []
        for index, message in enumerate(messages):
            content = message.get("content")
            if (
                message.get("role") != "tool"
                or message.get("name") not in _RETRIEVAL_TOOLS
                or str(content).startswith(_RETRIEVAL_RECEIPT)
            ):
                continue
            matched = False
            try:
                parsed = json.loads(content) if isinstance(content, str) else content
                matched = isinstance(parsed, dict) and bool(parsed.get("matches"))
            except (TypeError, json.JSONDecodeError):
                pass
            candidates.append((index, matched))
        if len(candidates) <= 1:
            return 0

        matched_indices = [index for index, matched in candidates if matched]
        keep = matched_indices[-1] if matched_indices else candidates[-1][0]
        compacted = 0
        for index, _ in candidates:
            if index == keep:
                continue
            messages[index]["content"] = ContextManager._retrieval_receipt(messages[index])
            ContextManager._strip_retrieval_assistant_payload(messages, index)
            compacted += 1
        return compacted

    @staticmethod
    def _strip_retrieval_assistant_payload(
        messages: list[dict[str, Any]], tool_index: int,
    ) -> None:
        tool_call_id = str(messages[tool_index].get("tool_call_id", ""))
        if not tool_call_id:
            return
        for index in range(tool_index - 1, -1, -1):
            assistant = messages[index]
            if assistant.get("role") != "assistant":
                continue
            call_ids = {
                str(call.get("id", "")) for call in assistant.get("tool_calls") or []
                if isinstance(call, dict)
            }
            if tool_call_id in call_ids:
                assistant["content"] = ""
                assistant.pop("reasoning_content", None)
                assistant.pop("thinking_blocks", None)
                return

    @staticmethod
    def compact_retrieval_history_for_persistence(messages: list[dict[str, Any]]) -> int:
        """Remove retrieval bodies before session persistence or summary input."""
        compacted = 0
        for index, message in enumerate(messages):
            if (
                message.get("role") == "tool"
                and message.get("name") in _RETRIEVAL_TOOLS
                and not str(message.get("content", "")).startswith(_RETRIEVAL_RECEIPT)
            ):
                message["content"] = ContextManager._retrieval_receipt(
                    message, retain_excerpt=True,
                )
                ContextManager._strip_retrieval_assistant_payload(messages, index)
                compacted += 1
        return compacted

    @staticmethod
    def _summary_index(messages: list[dict[str, Any]]) -> int | None:
        for idx, message in enumerate(messages):
            if message.get("role") == "system" and str(message.get("content", "")).startswith(SUMMARY_HEADING):
                return idx
        return None

    @staticmethod
    def _system_prefix_end(messages: list[dict[str, Any]]) -> int:
        """Return the first conversational message after protected system context."""
        index = 0
        while index < len(messages) and messages[index].get("role") == "system":
            index += 1
        return index

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
        episodic_memory: RetrievalResult | None = None,
    ) -> PreparedContext:
        work = messages
        compacted_retrieval = self.compact_transient_retrieval_results(work)
        offloaded = self.offload_tool_results(work, session_key) if session_key else 0
        estimated, source = self._estimate(work, tools)
        prepared = PreparedContext(
            work, estimated, source, offloaded_artifacts=offloaded,
            compacted_retrieval_results=compacted_retrieval,
            episodic_entries=len(episodic_memory.entries) if episodic_memory else 0,
            episodic_injected_chars=episodic_memory.injected_chars if episodic_memory else 0,
            episodic_candidates=episodic_memory.candidate_count if episodic_memory else 0,
            episodic_duplicates_suppressed=(
                episodic_memory.skipped_duplicates if episodic_memory else 0
            ),
        )
        if compacted_retrieval:
            prepared.actions.append("retrieval_compacted")
        soft_limit = int(self.effective_budget * self.policy.soft_threshold)
        target = int(self.effective_budget * self.policy.compaction_target)
        hard_limit = int(self.effective_budget * self.policy.hard_threshold)

        if estimated > soft_limit and session is not None:
            compact_event = None
            if self.hook_manager:
                compact_event = await self.hook_manager.dispatch(LifecycleEvent(
                    LifecycleEventType.CONTEXT_COMPACT,
                    payload={"messages": work, "estimated_tokens": estimated, "soft_limit": soft_limit, "target": target, "session": session},
                    session_key=session_key,
                ))
                if compact_event.decision and compact_event.decision.action is HookAction.DENY:
                    prepared.actions.append("compaction_denied")
                else:
                    work = compact_event.payload.get("messages", work)
            denied = bool(
                compact_event
                and compact_event.decision
                and compact_event.decision.action is HookAction.DENY
            )
            if not denied:
                for _ in range(self.policy.max_compaction_rounds):
                    ranges = session.context_turn_ranges()
                    eligible = ranges[:-self.policy.recent_turns] if len(ranges) > self.policy.recent_turns else []
                    if not eligible or estimated <= target:
                        break
                    start, end = eligible[0][0], eligible[-1][1]
                    chunk = Session._to_llm_messages(session.messages[start:end])
                    self.compact_retrieval_history_for_persistence(chunk)
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
                    history_start = self._system_prefix_end(work)
                    del work[history_start:history_start + remove_count]
                    summary_message = {"role": "system", "content": f"{SUMMARY_HEADING}\n{summary}"}
                    if summary_idx is None:
                        work.insert(self._system_prefix_end(work), summary_message)
                    else:
                        work[summary_idx] = summary_message
                    prepared.compacted_turns += len(eligible)
                    prepared.actions.append("compacted")
                    estimated, source = self._estimate(work, tools)
                    break

            if self.hook_manager:
                await self.hook_manager.dispatch(LifecycleEvent(
                    LifecycleEventType.CONTEXT_COMPACT,
                    payload={"messages": work, "prepared": prepared, "estimated_tokens": estimated, "actions": prepared.actions},
                    phase="after", session_key=session_key,
                ))

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
            "compacted_retrieval_results": prepared.compacted_retrieval_results,
            "episodic_entries": prepared.episodic_entries,
            "episodic_injected_chars": prepared.episodic_injected_chars,
            "episodic_candidates": prepared.episodic_candidates,
            "episodic_duplicates_suppressed": prepared.episodic_duplicates_suppressed,
            "actions": list(prepared.actions),
        })
        return prepared
