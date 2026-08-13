"""Small real-provider benchmark for rolling context summaries."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from nanobot.agent.context_manager import ContextManagementPolicy, ContextManager
from nanobot.cli.commands import _make_provider
from nanobot.config.loader import load_config
from nanobot.session.artifacts import ToolArtifactStore
from nanobot.session.manager import Session

CASES = [
    ("goal_recall", "GOAL-AURORA-17", "What is the exact project goal code?"),
    ("constraint_recall", "CONSTRAINT-NO-SQLITE-29", "What exact implementation constraint was specified?"),
    ("decision_recall", "DECISION-BLUE-API-41", "What exact architecture decision was recorded?"),
    ("pending_recall", "PENDING-MIGRATION-53", "What exact pending work item remains?"),
    ("tool_conclusion_recall", "TOOL-CONCLUSION-OK-67", "What exact conclusion did the validation tool produce?"),
]


def _session(case_id: str, marker: str) -> Session:
    session = Session(key=f"live-context:{case_id}")
    session.messages.extend([
        {"role": "user", "content": f"Remember this task fact exactly: {marker}."},
        {"role": "assistant", "content": f"Recorded {marker}."},
    ])
    filler = "Context engineering discussion about interfaces, testing, compatibility, and rollout. " * 220
    for index in range(9):
        session.messages.extend([
            {"role": "user", "content": f"Historical turn {index}. {filler}"},
            {"role": "assistant", "content": f"Historical response {index}. Requirements remain unchanged."},
        ])
    session.messages.extend([
        {"role": "user", "content": "Recent turn: keep the current implementation stable."},
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": "Recent turn: prepare the final verification."},
        {"role": "assistant", "content": "Ready."},
    ])
    return session


async def run_case(provider: Any, case_id: str, marker: str, question: str) -> dict[str, Any]:
    session = _session(case_id, marker)
    messages = [
        {"role": "system", "content": "Answer using retained session facts. Be concise."},
        *session.get_context_history(),
        {"role": "user", "content": question},
    ]
    with tempfile.TemporaryDirectory(prefix="litebot-live-context-") as root:
        manager = ContextManager(
            provider=provider,
            model=provider.get_default_model(),
            context_window_tokens=16_384,
            policy=ContextManagementPolicy(
                recent_turns=2,
                output_reserve_tokens=2048,
                safety_margin_tokens=512,
                soft_threshold=0.80,
                hard_threshold=0.92,
                compaction_target=0.68,
            ),
            artifacts=ToolArtifactStore(Path(root)),
        )
        started = time.perf_counter()
        prepared = await manager.prepare(messages, [], session=session, session_key=session.key)
        response = await provider.chat_with_retry(
            messages=prepared.messages, tools=[], model=provider.get_default_model(),
            max_tokens=256, temperature=0,
        )
        output = response.content or ""
        return {
            "case_id": case_id,
            "passed": marker in output and marker in session.context_summary,
            "marker": marker,
            "summary_contains_marker": marker in session.context_summary,
            "output_contains_marker": marker in output,
            "compacted_turns": prepared.compacted_turns,
            "hard_truncated_turns": prepared.hard_truncated_turns,
            "prepared_tokens": prepared.estimated_tokens,
            "token_source": prepared.token_source,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "usage": response.usage or {},
            "output": output,
        }


async def run(output: Path | None = None) -> dict[str, Any]:
    config = load_config()
    provider = _make_provider(config)
    results = []
    for case in CASES:
        result = await run_case(provider, *case)
        results.append(result)
        print(f"{'PASSED' if result['passed'] else 'FAILED'}  {result['case_id']} ({result['latency_ms']:.1f} ms)")
    report = {
        "schema_version": "litebot-live-context-benchmark/v1",
        "provider": config.get_provider_name(),
        "model": provider.get_default_model(),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "cases": results,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Live rolling-summary context benchmark")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    return 0 if asyncio.run(run(args.output))["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
