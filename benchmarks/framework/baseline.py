from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

BASELINES_ROOT = Path(__file__).resolve().parents[1] / "baselines"


class BaselineRegistry:
    def __init__(self, root: Path = BASELINES_ROOT):
        self.root = root
        self.registry_path = root / "registry.json"

    def _registry(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"schema_version": "litebot-baseline-registry/v1", "defaults": {}}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def promote(self, run: Path, profile: str, name: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", name):
            raise ValueError("invalid baseline name")
        source = run.resolve()
        for required in ("results.jsonl", "summary.json", "manifest.json"):
            if not (source / required).is_file():
                raise ValueError(f"run is missing {required}")
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        if summary.get("profile") != profile:
            raise ValueError("run profile does not match baseline profile")
        target = self.root / profile / name
        if target.exists():
            raise FileExistsError(f"baseline already exists: {profile}/{name}")
        target.mkdir(parents=True)
        for filename in ("results.jsonl", "summary.json", "manifest.json"):
            shutil.copy2(source / filename, target / filename)
        return target

    def set_default(self, profile: str, name: str) -> None:
        if not (self.root / profile / name / "results.jsonl").is_file():
            raise ValueError(f"unknown baseline: {profile}/{name}")
        registry = self._registry()
        registry.setdefault("defaults", {})[profile] = name
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    def resolve(self, reference: str) -> Path:
        parts = reference.split("/", 1)
        if len(parts) != 2:
            path = Path(reference)
            if path.exists():
                return path
            raise ValueError("baseline reference must be profile/name or a path")
        profile, name = parts
        if name == "default":
            name = self._registry().get("defaults", {}).get(profile)
            if not name:
                raise ValueError(f"no default baseline for profile {profile}")
        target = self.root / profile / name
        if not target.is_dir():
            raise ValueError(f"unknown baseline: {profile}/{name}")
        return target
