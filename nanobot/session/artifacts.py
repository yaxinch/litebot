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
_DISTINCTIVE_IDENTIFIER = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){2,}\b")


@dataclass(slots=True)
class ToolArtifact:
    artifact_id: str
    session_key: str
    tool_name: str
    tool_call_id: str
    content_type: str
    encoding: str
    size_bytes: int
    line_count: int | None
    sha256: str
    created_at: str
    relative_path: str


class ToolArtifactStore:
    """Write and read tool artifacts without exposing arbitrary paths."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.root = ensure_dir(workspace / "sessions" / "artifacts")

    @staticmethod
    def validate_artifact_id(artifact_id: str) -> None:
        if not _ARTIFACT_ID.fullmatch(artifact_id):
            raise ValueError("invalid artifact_id")

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
            line_count=payload.decode("utf-8").count("\n") + (1 if payload else 0),
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

    def _load_text(self, session_key: str, artifact_id: str) -> tuple[str, dict[str, Any]]:
        self.validate_artifact_id(artifact_id)
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
        encoding = str(metadata.get("encoding", "utf-8"))
        content_type = str(metadata.get("content_type", "text/plain"))
        if encoding.lower().replace("_", "-") != "utf-8" or not (
            content_type.startswith("text/") or content_type == "application/json"
        ):
            raise ValueError("artifact is not supported UTF-8 text")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("artifact is not valid UTF-8 text") from exc
        return text, metadata

    def get(self, session_key: str, artifact_id: str, offset: int, limit: int) -> dict[str, Any]:
        if offset < 0 or limit <= 0:
            raise ValueError("offset must be >= 0 and limit must be > 0")
        text, metadata = self._load_text(session_key, artifact_id)
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

    def search(
        self, session_key: str, artifact_id: str, query: str,
        max_matches: int = 5, context_chars: int = 500,
    ) -> dict[str, Any]:
        """Search a session-owned UTF-8 text artifact without returning the full value."""
        if not query:
            raise ValueError("query must not be empty")
        if len(query) > 256:
            raise ValueError("query exceeds 256 characters")
        if not 1 <= max_matches <= 5:
            raise ValueError("max_matches must be between 1 and 5")
        if not 0 <= context_chars <= 500:
            raise ValueError("context_chars must be between 0 and 500")
        result = self.search_many(
            session_key, artifact_id, [query], max_matches=max_matches,
            context_chars=context_chars, total_snippet_chars=max_matches * (2 * context_chars + len(query)),
        )
        result["query"] = query
        return result

    def search_many(
        self, session_key: str, artifact_id: str, queries: list[str],
        max_matches: int = 5, context_chars: int = 500,
        total_snippet_chars: int = 2_000,
    ) -> dict[str, Any]:
        """Search several literals with one artifact load and merge overlapping snippets."""
        if not queries or len(queries) > 5:
            raise ValueError("queries must contain between 1 and 5 items")
        if any(not query or len(query) > 256 for query in queries):
            raise ValueError("each query must contain between 1 and 256 characters")
        if not 1 <= max_matches <= 5:
            raise ValueError("max_matches must be between 1 and 5")
        if not 0 <= context_chars <= 500:
            raise ValueError("context_chars must be between 0 and 500")
        if total_snippet_chars <= 0:
            raise ValueError("total_snippet_chars must be positive")

        text, metadata = self._load_text(session_key, artifact_id)
        raw_hits: list[dict[str, Any]] = []
        # One extra hit per query is sufficient to report truncation while keeping
        # memory bounded for repetitive artifacts.
        for query in queries:
            cursor = 0
            found_count = 0
            while found_count <= max_matches:
                offset = text.find(query, cursor)
                if offset < 0:
                    break
                end = offset + len(query)
                raw_hits.append({
                    "query": query,
                    "offset": offset,
                    "end": end,
                    "snippet_start": max(0, offset - context_chars),
                    "snippet_end": min(len(text), end + context_chars),
                })
                found_count += 1
                cursor = offset + max(1, len(query))

        merged: list[dict[str, Any]] = []
        for hit in sorted(raw_hits, key=lambda item: (item["snippet_start"], item["snippet_end"])):
            if merged and hit["snippet_start"] <= merged[-1]["snippet_end"]:
                region = merged[-1]
                region["snippet_end"] = max(region["snippet_end"], hit["snippet_end"])
                region["query_matches"].append({
                    "query": hit["query"], "offset": hit["offset"], "end": hit["end"],
                })
            else:
                merged.append({
                    "snippet_start": hit["snippet_start"],
                    "snippet_end": hit["snippet_end"],
                    "query_matches": [{
                        "query": hit["query"], "offset": hit["offset"], "end": hit["end"],
                    }],
                })

        def rank(region: dict[str, Any]) -> tuple[int, int, int, int]:
            snippet = text[region["snippet_start"]:region["snippet_end"]]
            distinct_queries = {item["query"] for item in region["query_matches"]}
            return (
                -int(bool(_DISTINCTIVE_IDENTIFIER.search(snippet))),
                -len(distinct_queries),
                -max(len(query) for query in distinct_queries),
                min(item["offset"] for item in region["query_matches"]),
            )

        selected: list[dict[str, Any]] = []
        remaining = total_snippet_chars
        for region in sorted(merged, key=rank):
            if len(selected) >= max_matches or remaining <= 0:
                break
            query_matches = sorted(
                region["query_matches"], key=lambda item: (item["offset"], -len(item["query"])),
            )
            anchor = max(
                query_matches,
                key=lambda item: (len(item["query"]), -queries.index(item["query"]), -item["offset"]),
            )
            anchor_start = anchor["offset"]
            anchor_end = anchor["end"]
            minimum = anchor_end - anchor_start
            if minimum > remaining:
                continue
            desired_start = region["snippet_start"]
            desired_end = region["snippet_end"]
            desired_length = desired_end - desired_start
            if desired_length > remaining:
                spare = remaining - minimum
                before = min(anchor_start - desired_start, spare // 2)
                after = min(desired_end - anchor_end, spare - before)
                unused = spare - before - after
                before += min(anchor_start - desired_start - before, unused)
                desired_start = anchor_start - before
                desired_end = anchor_end + after
                query_matches = [
                    item for item in query_matches
                    if desired_start <= item["offset"] and item["end"] <= desired_end
                ]
            snippet = text[desired_start:desired_end]
            distinct_queries = list(dict.fromkeys(item["query"] for item in query_matches))
            representative = max(distinct_queries, key=lambda item: (len(item), -queries.index(item)))
            representative_match = next(
                item for item in query_matches if item["query"] == representative
            )
            selected.append({
                "query": representative,
                "queries": distinct_queries,
                "query_matches": query_matches,
                "offset": representative_match["offset"],
                "end": representative_match["end"],
                "snippet_start": desired_start,
                "snippet_end": desired_end,
                "snippet": snippet,
            })
            remaining -= len(snippet)

        return {
            "status": "matches" if selected else "no_match",
            "artifact_id": artifact_id,
            "queries": queries,
            "content_type": metadata.get("content_type", "text/plain"),
            "total_chars": len(text),
            "matches": selected,
            "has_more_matches": len(merged) > len(selected),
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
