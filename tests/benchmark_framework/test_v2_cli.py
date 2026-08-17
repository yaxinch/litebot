import json

from benchmarks.framework.cli import main


def test_validate_and_list_cli(capsys):
    assert main(["validate"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["core_cases"] == 72
    assert main(["list", "--category", "tool_safety"]) == 0
    assert "Total: 12" in capsys.readouterr().out


def test_single_case_run_has_unified_result(tmp_path):
    output = tmp_path / "run"
    assert main(["run", "--case", "tool_reliability.timeout", "--output", str(output)]) == 0
    row = json.loads((output / "results.jsonl").read_text(encoding="utf-8"))
    required = {
        "case_id", "category", "passed", "prompt_tokens", "completion_tokens",
        "total_tokens", "latency", "tool_rounds", "tool_calls", "tool_trace",
        "context_compressions", "memory_hits", "tool_denied", "retry_count",
        "failure_category", "verifier_reason",
    }
    assert required <= row.keys()
    assert row["passed"] is True
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["git_commit"]) in {40, 64} or summary["git_commit"] == "unknown"


def test_live_and_judge_are_explicitly_gated():
    assert main(["run", "--profile", "live"]) == 2
    assert main(["run", "--profile", "judge"]) == 2
