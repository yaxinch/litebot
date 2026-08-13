import pytest

from benchmarks.context_management import measure


@pytest.mark.asyncio
async def test_large_context_benchmark_reduces_tokens_and_recovers_artifact():
    result = await measure(100 * 1024)
    assert result["token_reduction_percent"] >= 60
    assert result["offloaded_artifacts"] == 1
    assert result["artifact_recovered"] is True
    assert result["tool_structure_valid"] is True
