from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context_summary import ContextSummarizer
from nanobot.providers.base import LLMResponse


@pytest.mark.asyncio
async def test_summary_preserves_provider_output():
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="## Current Goal and Success Criteria\nship it"))
    summarizer = ContextSummarizer(provider, "test")
    result = await summarizer.summarize("", [{"role": "user", "content": "ship it"}])
    assert result == "## Current Goal and Success Criteria\nship it"


@pytest.mark.asyncio
async def test_summary_failure_returns_none():
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="", finish_reason="error"))
    assert await ContextSummarizer(provider, "test").summarize("old", []) is None
