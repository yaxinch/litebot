"""Deterministic offline benchmark for structured episodic retrieval."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from benchmarks.collector import BenchmarkCollector, RecordingProvider
from benchmarks.providers.scripted import ScriptedProvider
from nanobot.agent.episodic_memory import EpisodicMemorySource, EpisodicMemoryStore
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    case_id: str
    query: str
    expected: str | None
    entries: tuple[dict[str, Any], ...] = ()
    legacy_history: str = ""
    expected_top1: str | None = None
    end_to_end: bool = False


class MemoryAwareScriptedProvider(ScriptedProvider):
    """Offline provider that answers only when AgentLoop injects the target memory."""

    def __init__(self, target_memory: str, target_query: str):
        super().__init__([])
        self.target_memory = target_memory
        self.target_query = target_query
        self.saw_target_memory = False
        self.saw_target_query = False

    async def chat(self, messages, tools=None, model=None, **kwargs) -> LLMResponse:
        del tools, model, kwargs
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        self.saw_target_memory = self.target_memory in prompt
        self.saw_target_query = self.target_query in prompt
        answer = "SQLite" if self.saw_target_memory and self.saw_target_query else "MEMORY_NOT_FOUND"
        return LLMResponse(
            content=answer,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


def _entry(content: str, category: str = "fact", importance: int = 3, *, days_old: int = 0) -> dict[str, Any]:
    timestamp = datetime(2026, 8, 14, tzinfo=timezone.utc) - timedelta(days=days_old)
    return {
        "timestamp": timestamp.isoformat(),
        "category": category,
        "content": content,
        "importance": importance,
    }


def cases() -> tuple[RetrievalCase, ...]:
    distractors = (
        _entry("Discussed frontend colors and lunch plans."),
        _entry("The staging environment uses an unrelated cache."),
    )
    return (
        RetrievalCase(
            "exact_constraint", "Which database is forbidden in v1?", "vector database",
            distractors + (_entry("V1 must not use an external vector database.", "constraint", 5),),
        ),
        RetrievalCase(
            "cjk_constraint", "向量数据库有什么项目约束", "不要引入外部向量数据库",
            distractors + (_entry("第一版不要引入外部向量数据库", "constraint", 5),),
        ),
        RetrievalCase(
            "cross_session", "What deployment namespace was selected?", "litebot-prod",
            distractors + (_entry("Deployment namespace is litebot-prod.", "decision", 4),),
        ),
        RetrievalCase(
            "recency_importance", "release marker", "recent release marker",
            (
                _entry("recent release marker", importance=1, days_old=1),
                _entry("old release marker", importance=5, days_old=90),
            ),
        ),
        RetrievalCase(
            "duplicate_suppression", "alpha beta gamma", "alpha beta gamma delta",
            (
                _entry("alpha beta gamma delta", "decision", 5),
                _entry("alpha beta gamma delta.", "decision", 4),
                _entry("alpha independent record"),
            ),
        ),
        RetrievalCase(
            "legacy_migration", "What was the legacy marker?", "LEGACY-778",
            legacy_history="[2026-08-01 12:00] Historical decision marker LEGACY-778\n\n",
        ),
        RetrievalCase(
            "zero_match", "unfindable-zeta-token", None,
            distractors,
        ),
        RetrievalCase(
            "conflict_update", "What deployment namespace is currently selected?", "litebot-prod",
            (
                _entry("Deployment namespace is litebot-staging.", "decision", 3, days_old=90),
                _entry("Deployment namespace is litebot-prod.", "decision", 3),
            ),
            expected_top1="Deployment namespace is litebot-prod.",
        ),
        RetrievalCase(
            "agent_memory_recall",
            "Which database should Project Atlas use for local persistence?",
            "SQLite",
            (_entry("Project Atlas must use SQLite for local persistence.", "constraint", 5),),
            expected_top1="Project Atlas must use SQLite for local persistence.",
            end_to_end=True,
        ),
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


async def _run_agent_memory_recall(
    workspace: Path, case: RetrievalCase, collector: BenchmarkCollector,
) -> tuple[str, bool, dict[str, int]]:
    if not case.expected_top1:
        raise ValueError("end-to-end memory case requires expected_top1")
    inner = MemoryAwareScriptedProvider(case.expected_top1, case.query)
    provider = RecordingProvider(inner, collector)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=workspace,
        context_window_tokens=10_000,
        restrict_to_workspace=True,
    )
    try:
        outbound = await loop.process_direct(
            case.query,
            session_key="benchmark:atlas-query",
            channel="benchmark",
            chat_id="atlas-query",
        )
        answer = outbound.content if outbound else ""
    finally:
        await loop.close_mcp()
        loop.stop()
    injection_success = inner.saw_target_memory and inner.saw_target_query
    return answer, injection_success, collector.usage


def run_benchmark(*, repetitions: int = 20) -> dict[str, Any]:
    case_results: list[dict[str, Any]] = []
    latencies: list[float] = []
    reciprocal_ranks: list[float] = []
    recall_at_1 = recall_at_3 = 0
    expected_cases = 0
    duplicate_rates: list[float] = []

    for case in cases():
        with tempfile.TemporaryDirectory(prefix=f"litebot-memory-{case.case_id}-") as raw_workspace:
            workspace = Path(raw_workspace)
            store = EpisodicMemoryStore(workspace)
            if case.legacy_history:
                store.history_file.write_text(case.legacy_history, encoding="utf-8")
            if case.entries:
                store.append(
                    case.entries,
                    source=EpisodicMemorySource(
                        kind="benchmark",
                        session_key=("benchmark:atlas-seed" if case.end_to_end else "benchmark:seed"),
                    ),
                )
            query_session = "benchmark:atlas-query" if case.end_to_end else "benchmark:query"
            result = store.retrieve(
                case.query, session_key=query_session,
                now=datetime(2026, 8, 14, tzinfo=timezone.utc),
            )
            for _ in range(repetitions):
                started = time.perf_counter()
                store.retrieve(
                    case.query, session_key=query_session,
                    now=datetime(2026, 8, 14, tzinfo=timezone.utc),
                )
                latencies.append((time.perf_counter() - started) * 1000)

            contents = [item.entry.content for item in result.entries]
            rank = next(
                (index for index, content in enumerate(contents, 1) if case.expected and case.expected in content),
                None,
            )
            top1_memory = contents[0] if contents else None
            if case.expected is not None:
                expected_cases += 1
                recall_at_1 += int(rank == 1)
                recall_at_3 += int(rank is not None and rank <= 3)
                reciprocal_ranks.append(1 / rank if rank else 0.0)
                retrieval_success = rank is not None and rank <= 3
            else:
                retrieval_success = not result.entries
            if case.expected_top1 is not None:
                retrieval_success = retrieval_success and top1_memory == case.expected_top1

            task_success: bool | None = None
            token_usage: dict[str, int] | None = None
            injection_success: bool | None = None
            final_answer: str | None = None
            if case.end_to_end:
                collector = BenchmarkCollector()
                final_answer, injection_success, token_usage = asyncio.run(
                    _run_agent_memory_recall(workspace, case, collector)
                )
                retrieval_success = retrieval_success and injection_success
                task_success = bool(case.expected and case.expected in final_answer)
            strict_success = retrieval_success and (task_success if task_success is not None else True)
            passed = strict_success

            new_memory_rank = next(
                (index for index, content in enumerate(contents, 1) if content == "Deployment namespace is litebot-prod."),
                None,
            ) if case.case_id == "conflict_update" else None
            old_memory_rank = next(
                (index for index, content in enumerate(contents, 1) if content == "Deployment namespace is litebot-staging."),
                None,
            ) if case.case_id == "conflict_update" else None
            duplicate_rate = (
                result.skipped_duplicates / result.candidate_count
                if result.candidate_count else 0.0
            )
            duplicate_rates.append(duplicate_rate)
            case_results.append({
                "case_name": case.case_id,
                "case_id": case.case_id,
                "passed": passed,
                "retrieval_success": retrieval_success,
                "task_success": task_success,
                "strict_success": strict_success,
                "expected": case.expected,
                "expected_top1": case.expected_top1,
                "actual_top1": top1_memory,
                "top1_memory": top1_memory,
                "rank": rank,
                "new_memory_rank": new_memory_rank,
                "old_memory_rank": old_memory_rank,
                "candidate_count": result.candidate_count,
                "returned_count": len(result.entries),
                "returned_memory_count": len(result.entries),
                "duplicates_suppressed": result.skipped_duplicates,
                "injected_chars": result.injected_chars,
                "scores": [round(item.relevance_score, 6) for item in result.entries],
                "injection_success": injection_success,
                "final_answer": final_answer,
                "token_usage": token_usage,
            })

    return {
        "schema_version": "litebot-episodic-benchmark/v1",
        "baseline": {"description": "no automatic episodic injection", "recall_at_1": 0.0, "recall_at_3": 0.0, "mrr": 0.0},
        "candidate": {
            "recall_at_1": recall_at_1 / expected_cases,
            "recall_at_3": recall_at_3 / expected_cases,
            "mrr": statistics.fmean(reciprocal_ranks),
            "mean_duplicate_rate": statistics.fmean(duplicate_rates),
            "mean_injected_chars": statistics.fmean(item["injected_chars"] for item in case_results),
            "retrieval_latency_ms_p50": _percentile(latencies, 0.50),
            "retrieval_latency_ms_p95": _percentile(latencies, 0.95),
        },
        "counts": {
            "total": len(case_results),
            "passed": sum(item["passed"] for item in case_results),
            "failed": sum(not item["passed"] for item in case_results),
        },
        "cases": case_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiteBot deterministic episodic-memory benchmark")
    parser.add_argument("--output")
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args(argv)
    report = run_benchmark(repetitions=max(1, args.repetitions))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Results: {output}")
    else:
        print(rendered)
    return int(report["counts"]["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
