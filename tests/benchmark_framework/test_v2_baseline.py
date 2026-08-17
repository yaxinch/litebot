import json

import pytest

from benchmarks.framework.baseline import BaselineRegistry


def test_named_baseline_promote_default_and_resolve(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({"profile": "core"}), encoding="utf-8")
    registry = BaselineRegistry(tmp_path / "baselines")
    promoted = registry.promote(run, "core", "phase4")
    registry.set_default("core", "phase4")
    assert registry.resolve("core/default") == promoted
    with pytest.raises(FileExistsError):
        registry.promote(run, "core", "phase4")
