from benchmarks.episodic_memory import run_benchmark


def test_deterministic_episodic_memory_benchmark_passes() -> None:
    report = run_benchmark(repetitions=1)

    assert report["counts"] == {"total": 9, "passed": 9, "failed": 0}
    assert [item["case_id"] for item in report["cases"]] == [
        "exact_constraint",
        "cjk_constraint",
        "cross_session",
        "recency_importance",
        "duplicate_suppression",
        "legacy_migration",
        "zero_match",
        "conflict_update",
        "agent_memory_recall",
    ]
    assert report["baseline"]["recall_at_3"] == 0
    assert report["candidate"]["recall_at_3"] == 1
    for case in report["cases"]:
        assert {
            "case_name", "passed", "retrieval_success", "task_success",
            "strict_success", "returned_memory_count", "top1_memory", "token_usage",
        } <= case.keys()
    duplicate = next(item for item in report["cases"] if item["case_id"] == "duplicate_suppression")
    assert duplicate["duplicates_suppressed"] == 1

    conflict = next(item for item in report["cases"] if item["case_id"] == "conflict_update")
    assert conflict["strict_success"] is True
    assert conflict["actual_top1"] == "Deployment namespace is litebot-prod."
    assert conflict["new_memory_rank"] == 1
    assert conflict["old_memory_rank"] > conflict["new_memory_rank"]

    agent = next(item for item in report["cases"] if item["case_id"] == "agent_memory_recall")
    assert agent["retrieval_success"] is True
    assert agent["task_success"] is True
    assert agent["strict_success"] is True
    assert agent["top1_memory"] == "Project Atlas must use SQLite for local persistence."
    assert agent["final_answer"] == "SQLite"
    assert agent["token_usage"]["total_tokens"] > 0
