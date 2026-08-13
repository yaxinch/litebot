"""Persistent, session-isolated storage for large tool results."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from nanobot.utils.helpers import ensure_dir, safe_filename

_ARTIFACT_ID = re.compile(r"^[a-f0-9]{32}$")


@dataclass(slots=True)
class ToolArtifact:
    artifact_id: str
    session_key: str
    tool_name: str
    tool_call_id: str
    content_type: str
    encoding: str
    size_bytes: int
    sha256: str
    created_at: str
    relative_path: str


class ToolArtifactStore:
    """Write and read tool artifacts without exposing arbitrary paths."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.root = ensure_dir(workspace / "sessions" / "artifacts")

    @staticmethod
    def serialize(value: Any) -> tuple[bytes, str]:
        if isinstance(value, str):
            return value.encode("utf-8"), "text/plain"
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"), "application/json"
        except (TypeError, ValueError):
            return str(value).encode("utf-8"), "text/plain"

    def _session_dir(self, session_key: str, *, create: bool = True) -> Path:
        path = self.root / safe_filename(session_key.replace(":", "_"))
        return ensure_dir(path) if create else path

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def put(self, session_key: str, tool_name: str, tool_call_id: str, value: Any) -> ToolArtifact:
        payload, content_type = self.serialize(value)
        artifact_id = uuid.uuid4().hex
        directory = self._session_dir(session_key)
        data_path = directory / f"{artifact_id}.data"
        meta_path = directory / f"{artifact_id}.json"
        artifact = ToolArtifact(
            artifact_id=artifact_id,
            session_key=session_key,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            content_type=content_type,
            encoding="utf-8",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            created_at=datetime.now().isoformat(),
            relative_path=data_path.relative_to(self.workspace).as_posix(),
        )
        self._atomic_write(data_path, payload)
        try:
            self._atomic_write(
                meta_path,
                json.dumps(asdict(artifact), ensure_ascii=False, indent=2).encode("utf-8"),
            )
        except Exception:
            data_path.unlink(missing_ok=True)
            raise
        return artifact

    def get(self, session_key: str, artifact_id: str, offset: int, limit: int) -> dict[str, Any]:
        if not _ARTIFACT_ID.fullmatch(artifact_id):
            raise ValueError("invalid artifact_id")
        if offset < 0 or limit <= 0:
            raise ValueError("offset must be >= 0 and limit must be > 0")
        directory = self._session_dir(session_key, create=False)
        meta_path = directory / f"{artifact_id}.json"
        data_path = directory / f"{artifact_id}.data"
        if not meta_path.is_file() or not data_path.is_file():
            raise FileNotFoundError("artifact not found for this session")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("session_key") != session_key:
            raise PermissionError("artifact does not belong to this session")
        payload = data_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata.get("sha256"):
            raise ValueError("artifact checksum mismatch")
        text = payload.decode(metadata.get("encoding", "utf-8"))
        if offset > len(text):
            raise ValueError("offset exceeds artifact length")
        end = min(len(text), offset + limit)
        return {
            "artifact_id": artifact_id,
            "content": text[offset:end],
            "offset": offset,
            "end": end,
            "total_chars": len(text),
            "has_more": end < len(text),
            "content_type": metadata.get("content_type", "text/plain"),
        }

    def collect_garbage(
        self, ttl_days: int, protected_artifact_ids: set[str] | None = None,
        *, now: datetime | None = None,
    ) -> dict[str, int]:
        """Delete expired artifact pairs unless referenced by an active session."""
        protected = protected_artifact_ids or set()
        cutoff = (now or datetime.now()) - timedelta(days=ttl_days)
        deleted = protected_count = invalid = 0
        for meta_path in self.root.glob("*/*.json"):
            artifact_id = meta_path.stem
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                created_at = datetime.fromisoformat(metadata["created_at"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                invalid += 1
                continue
            if created_at >= cutoff:
                continue
            if artifact_id in protected:
                protected_count += 1
                continue
            data_path = meta_path.with_suffix(".data")
            data_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            deleted += 1
            try:
                meta_path.parent.rmdir()
            except OSError:
                pass
        return {"deleted": deleted, "protected": protected_count, "invalid": invalid}
