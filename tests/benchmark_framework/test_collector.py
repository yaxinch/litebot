import pytest

from benchmarks.collector import BenchmarkCollector, RecordingProvider, RecordingTool, redact
from benchmarks.providers.scripted import ScriptedProvider
from benchmarks.tools.fixtures import ErrorTool


@pytest.mark.asyncio
async def test_provider_usage_is_accumulated():
    collector = BenchmarkCollector()
    provider = RecordingProvider(ScriptedProvider([
        {"content": "one", "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        {"content": "two", "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}},
    ]), collector)
    await provider.chat(messages=[])
    await provider.chat(messages=[])
    assert collector.usage == {"prompt_tokens": 7, "completion_tokens": 10, "total_tokens": 17}


@pytest.mark.asyncio
async def test_tool_error_result_and_redaction():
    collector = BenchmarkCollector()
    tool = RecordingTool(ErrorTool(), collector)
    result = await tool.execute(reason="planned")
    assert result.startswith("Error")
    assert collector.tool_errors == 1
    assert collector.tool_events[0]["status"] == "error_result"
    assert redact({"api_key": "secret", "value": "ok"}) == {"api_key": "[REDACTED]", "value": "ok"}

