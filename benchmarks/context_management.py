"""Deterministic baseline/candidate measurements for context management."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from nanobot.agent.context_manager import ContextManagementPolicy, ContextManager
from nanobot.session.artifacts import ToolArtifactStore
from nanobot.utils.helpers import estimate_prompt_tokens


def _tool_group(payload: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "Inspect the generated report."},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": "report-1", "type": "function",
                "function": {"name": "report", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "report-1", "name": "report", "content": payload},
        {"role": "user", "content": "Use the report to answer."},
    ]


async def measure(size_bytes: int) -> dict[str, Any]:
    marker = "CONTEXT-MARKER-7429"
    payload = "x" * (size_bytes // 2) + marker + "y" * (size_bytes - size_bytes // 2 - len(marker))
    baseline = [{"role": "system", "content": "system"}, *_tool_group(payload)]
    baseline_tokens = estimate_prompt_tokens(baseline, [])

    with tempfile.TemporaryDirectory(prefix="litebot-context-benchmark-") as root:
        workspace = Path(root)
        store = ToolArtifactStore(workspace)
        manager = ContextManager(
            provider=object(), model="offline", context_window_tokens=2_000_000,
            policy=ContextManagementPolicy(tool_offload_threshold_bytes=8192),
            artifacts=store,
        )
        candidate = [{**message} for message in baseline]
        offloaded = manager.offload_tool_results(candidate, "benchmark:context")
        candidate_tokens = estimate_prompt_tokens(candidate, [])
        placeholder = next(message["content"] for message in candidate if message.get("role") == "tool")
        artifact_id = placeholder.split("artifact_id: ", 1)[1].splitlines()[0]
        restored = store.get("benchmark:context", artifact_id, 0, len(payload))["content"]
        return {
            "size_bytes": size_bytes,
            "baseline_prompt_tokens": baseline_tokens,
            "candidate_prompt_tokens": candidate_tokens,
            "token_reduction": baseline_tokens - candidate_tokens,
            "token_reduction_percent": round((baseline_tokens - candidate_tokens) / baseline_tokens * 100, 2),
            "offloaded_artifacts": offloaded,
            "artifact_recovered": restored == payload,
            "marker_recovered": marker in restored,
            "tool_structure_valid": ContextManager.validate_tool_structure(candidate),
        }


async def run(output: Path | None = None) -> dict[str, Any]:
    cases = [await measure(size) for size in (100 * 1024, 1024 * 1024)]
    report = {
        "schema_version": "litebot-context-benchmark/v1",
        "cases": cases,
        "passed": all(
            case["artifact_recovered"]
            and case["marker_recovered"]
            and case["tool_structure_valid"]
            and case["token_reduction_percent"] >= 60
            for case in cases
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="LiteBot context management benchmark")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    return 0 if asyncio.run(run(args.output))["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
