"""Session-scoped rolling summaries used for prompt compaction."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider

SUMMARY_HEADING = "[Session Context Summary]"


class ContextSummarizer:
    """Replace an existing session summary with one covering an additional chunk."""

    def __init__(self, provider: LLMProvider, model: str):
        self.provider = provider
        self.model = model

    @staticmethod
    def _format(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "unknown")).upper()
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            tool_calls = message.get("tool_calls") or []
            suffix = f" tool_calls={json.dumps(tool_calls, ensure_ascii=False)}" if tool_calls else ""
            lines.append(f"{role}: {content}{suffix}")
        return "\n".join(lines)

    async def summarize(self, existing: str, messages: list[dict[str, Any]]) -> str | None:
        prompt = f"""Update the session context summary using the conversation chunk below.
Return only concise markdown with these exact headings:
## Current Goal and Success Criteria
## User Constraints and Preferences
## Completed Work
## Pending Work and Blockers
## Key Facts and Decisions
## Tool Conclusions and Artifact References

Preserve still-relevant information from the existing summary. Keep concrete identifiers,
decisions, unresolved tasks, tool conclusions, and artifact_id/path references. Do not invent facts.

Existing summary:
{existing or "(empty)"}

Conversation chunk:
{self._format(messages)}"""
        try:
            response = await self.provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": "You compact session context while preserving task state."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                tools=[],
            )
            summary = (response.content or "").strip()
            if response.finish_reason == "error" or not summary:
                return None
            return summary
        except Exception:
            logger.exception("Session context summarization failed")
            return None
