from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .models import Category, EvalCase, Profile

CASES_ROOT = Path(__file__).resolve().parents[1] / "cases" / "v2"


class CaseRegistry:
    def __init__(self, root: Path = CASES_ROOT):
        self.root = root

    def load(self, profile: Profile | str | None = None) -> list[EvalCase]:
        selected = Profile(profile) if profile else None
        cases: list[EvalCase] = []
        for path in sorted(self.root.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError(f"{path}: expected JSON array")
            cases.extend(EvalCase.model_validate(item) for item in raw)
        ids = [case.case_id for case in cases]
        duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate case ids: {', '.join(duplicates)}")
        if selected:
            cases = [case for case in cases if case.profile is selected]
        return sorted(cases, key=lambda case: case.case_id)

    def validate_core_inventory(self) -> dict[str, int]:
        core = self.load(Profile.CORE)
        counts = Counter(case.category.value for case in core)
        expected = {category.value: 12 for category in Category}
        if len(core) != 72 or dict(counts) != expected:
            raise ValueError(f"core inventory must be 72 cases (12/category), got {dict(counts)}")
        return expected

    @staticmethod
    def manifest_hash(cases: list[EvalCase]) -> str:
        payload = [case.model_dump(mode="json", exclude_none=True) for case in cases]
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()
