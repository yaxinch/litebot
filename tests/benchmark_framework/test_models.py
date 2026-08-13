import json

import pytest

from benchmarks.models import load_cases


def test_fixed_case_counts_and_unique_ids():
    from benchmarks.run import CASES_DIR
    expected = {"deterministic": 18, "integration": 4, "live": 6}
    ids = []
    for suite, count in expected.items():
        cases = load_cases(CASES_DIR / f"{suite}.json", suite)
        assert len(cases) == count
        ids.extend(case.id for case in cases)
    assert len(ids) == len(set(ids)) == 28


def test_case_loader_rejects_cross_suite(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{
        "schema_version": "litebot-benchmark-case/v1", "id": "live_x",
        "suite": "live", "category": "x", "description": "x",
        "turns": [{"content": "x"}], "assertions": []
    }]), encoding="utf-8")
    with pytest.raises(ValueError, match="another suite"):
        load_cases(path, "deterministic")


def test_live_suite_requires_explicit_gate():
    from benchmarks.run import main

    assert main(["run", "--suite", "live", "--case", "live_web_search"]) == 2
