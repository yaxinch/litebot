from pathlib import Path

import pytest

from benchmarks.models import load_cases
from benchmarks.run import CASES_DIR, run_case


@pytest.mark.asyncio
async def test_case_workspace_is_removed_and_repo_is_not_used(tmp_path, monkeypatch):
    case = load_cases(CASES_DIR / "deterministic.json", "deterministic")[0]
    created = []
    import benchmarks.run as module
    original = module.tempfile.mkdtemp

    def tracking(*args, **kwargs):
        value = original(dir=tmp_path, *args, **kwargs)
        created.append(Path(value))
        return value

    monkeypatch.setattr(module.tempfile, "mkdtemp", tracking)
    result = await run_case(case, "test-run", tmp_path / "artifacts", False)
    assert result["status"] == "PASSED"
    assert created and not created[0].exists()
