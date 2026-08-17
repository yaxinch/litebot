"""Central policy, redaction, audit, and error classification for tool execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from fnmatch import fnmatchcase
from pathlib import Path, PurePath
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from loguru import logger

from nanobot.config.schema import ToolPolicyConfig, ToolPolicyRuleConfig


class PolicyAction(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


class ToolOutcome(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class RetryClass(StrEnum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non-retryable"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"


@dataclass(slots=True)
class ToolRequestContext:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    schema: dict[str, Any] | None = None
    workspace: Path | None = None
    restrict_to_workspace: bool = False
    run_id: str | None = None
    session_key: str | None = None
    case_id: str | None = None
    attempt: int = 1


@dataclass(slots=True)
class ToolPolicyDecision:
    action: PolicyAction
    reason: str
    reason_codes: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)
    argument_hash: str = ""
    confirmation: str | None = None


@dataclass(slots=True)
class ToolErrorInfo:
    retry_class: RetryClass
    retryable: bool
    stage: str
    message: str
    exception_type: str | None = None
    status_code: int | None = None


@dataclass(slots=True)
class ToolAuditWriteResult:
    """Observable outcome of one audit persistence attempt."""

    succeeded: bool
    record: dict[str, Any]
    error: str | None = None


class SensitiveDataRedactor:
    """Recursively remove common credentials from structured and textual data."""

    _VALUE_PATTERNS = (
        re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+"),
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret)"
            r"(\s*[:=]\s*)[^\s,;]+"
        ),
        re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{12,}|AKIA[A-Z0-9]{16})\b"),
    )

    def __init__(self, keys: list[str] | None = None, replacement: str = "[REDACTED]"):
        self.keys = {self._normalize_key(key) for key in (keys or [])}
        self.replacement = replacement

    @staticmethod
    def _normalize_key(key: str) -> str:
        return re.sub(r"[^a-z0-9]", "", key.lower())

    def redact(self, value: Any, key: str | None = None) -> Any:
        normalized_key = self._normalize_key(key) if key is not None else ""
        if normalized_key and any(
            normalized_key == sensitive or normalized_key.endswith(sensitive)
            for sensitive in self.keys
        ):
            return self.replacement
        if isinstance(value, dict):
            return {str(k): self.redact(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        if isinstance(value, str):
            return self.redact_text(value)
        return value

    def redact_text(self, value: str) -> str:
        redacted = value
        try:
            parsed = urlsplit(redacted)
            if parsed.scheme and parsed.hostname:
                host = parsed.hostname
                if parsed.port:
                    host += f":{parsed.port}"
                user = parsed.username or ""
                credentials = (
                    f"{user}:{self.replacement}@" if parsed.password is not None else
                    f"{user}@" if parsed.username is not None else ""
                )
                query = urlencode([
                    (key, self.replacement if self._normalize_key(key) in self.keys else item)
                    for key, item in parse_qsl(parsed.query, keep_blank_values=True)
                ])
                redacted = urlunsplit(
                    (parsed.scheme, f"{credentials}{host}", parsed.path, query, parsed.fragment)
                )
        except ValueError:
            pass
        redacted = self._VALUE_PATTERNS[0].sub(rf"\1{self.replacement}", redacted)
        redacted = self._VALUE_PATTERNS[1].sub(rf"\1\2{self.replacement}", redacted)
        redacted = self._VALUE_PATTERNS[2].sub(self.replacement, redacted)
        return redacted


class ToolErrorClassifier:
    """Normalize exceptions and legacy error strings into retry guidance."""

    @classmethod
    def classify(
        cls,
        error: BaseException | str | None,
        *,
        stage: str = "execution",
        status_code: int | None = None,
    ) -> ToolErrorInfo:
        message = "" if error is None else str(error)
        lower = message.lower()
        exception_type = type(error).__name__ if isinstance(error, BaseException) else None

        if stage in {"lookup", "validation"}:
            category, retryable = RetryClass.INVALID_REQUEST, False
        elif status_code == 429 or "429" in lower or "rate limit" in lower:
            category, retryable = RetryClass.RATE_LIMIT, True
        elif isinstance(error, (asyncio.TimeoutError, TimeoutError)) or any(
            marker in lower for marker in ("timeout", "timed out")
        ):
            category, retryable = RetryClass.TIMEOUT, True
        elif status_code is not None and 500 <= status_code < 600:
            category, retryable = RetryClass.RETRYABLE, True
        elif any(
            marker in lower
            for marker in (
                "connection reset", "connection refused", "temporarily unavailable",
                "server unavailable", "502", "503", "504",
            )
        ):
            category, retryable = RetryClass.RETRYABLE, True
        else:
            category, retryable = RetryClass.NON_RETRYABLE, False
        return ToolErrorInfo(category, retryable, stage, message, exception_type, status_code)


class ToolAuditSink:
    """Append-only, redacted JSONL audit sink rooted inside the workspace."""

    def __init__(
        self,
        workspace: Path,
        relative_path: str,
        redactor: SensitiveDataRedactor,
        *,
        line_writer: Callable[[str], None] | None = None,
    ):
        self.workspace = workspace.expanduser().resolve()
        candidate = (self.workspace / relative_path).resolve()
        if not self._is_under(candidate, self.workspace):
            raise ValueError("tool policy auditPath must stay inside the workspace")
        self.path = candidate
        self.redactor = redactor
        self._lock = asyncio.Lock()
        self._line_writer = line_writer or self._append

    @staticmethod
    def _is_under(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    async def write(self, record: dict[str, Any]) -> ToolAuditWriteResult:
        clean = self.redactor.redact(record)
        clean.setdefault("timestamp", datetime.now(UTC).isoformat())
        line = json.dumps(clean, ensure_ascii=False, default=str) + "\n"
        try:
            async with self._lock:
                await asyncio.to_thread(self._line_writer, line)
        except Exception as exc:
            error = self.redactor.redact_text(f"{type(exc).__name__}: {exc}")
            logger.error("Tool audit write failed: {}", error)
            return ToolAuditWriteResult(False, clean, error)
        return ToolAuditWriteResult(True, clean)

    def _append(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)


class ToolPolicyEngine:
    """Tool-agnostic runtime policy evaluator."""

    _DANGEROUS_COMMANDS = (
        re.compile(r"\brm\s+-[rf]{1,2}\b", re.I),
        re.compile(r"\bdel\s+/[fq]\b", re.I),
        re.compile(r"\brmdir\s+/s\b", re.I),
        re.compile(r"(?:^|[;&|]\s*)format\b", re.I),
        re.compile(r"\b(mkfs|diskpart)\b", re.I),
        re.compile(r"\bdd\s+if=", re.I),
        re.compile(r">\s*/dev/sd", re.I),
        re.compile(r"\b(shutdown|reboot|poweroff)\b", re.I),
        re.compile(r":\(\)\s*\{.*\};\s*:", re.I),
    )
    _URL_RE = re.compile(r"https?://[^\s\"'`;|<>]+", re.I)

    def __init__(
        self,
        config: ToolPolicyConfig | None,
        workspace: Path,
        *,
        restrict_to_workspace: bool = False,
        extra_read_roots: list[Path] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config or ToolPolicyConfig()
        self.workspace = workspace.expanduser().resolve()
        self.restrict_to_workspace = restrict_to_workspace
        self.extra_read_roots = [
            root.expanduser().resolve() for root in (extra_read_roots or [])
        ]
        self._clock = clock
        self.redactor = SensitiveDataRedactor(
            self.config.redaction.keys, self.config.redaction.replacement
        )
        self.audit = ToolAuditSink(self.workspace, self.config.audit_path, self.redactor)
        self._seen: dict[tuple[str, str, str], float] = {}
        self._seen_lock = asyncio.Lock()
        self._rules = sorted(
            enumerate(self.config.rules), key=lambda item: (-item[1].priority, item[0])
        )

    @staticmethod
    def fingerprint(name: str, arguments: dict[str, Any]) -> str:
        canonical = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(f"{name}\0{canonical}".encode()).hexdigest()

    async def evaluate(self, context: ToolRequestContext) -> ToolPolicyDecision:
        signature = self.fingerprint(context.name, context.arguments)
        if not self.config.enabled:
            return ToolPolicyDecision(PolicyAction.ALLOW, "tool policy disabled", ["policy.disabled"], argument_hash=signature)

        decisions: list[tuple[PolicyAction, str, str, str | None]] = []
        for _, rule in self._rules:
            if self._matches(rule, context):
                decisions.append((PolicyAction(rule.action), rule.reason, f"rule.{rule.id}", rule.id))

        decisions.extend(self._path_decisions(context))
        decisions.extend(self._command_decisions(context))
        duplicate = await self._duplicate_decision(context, signature)
        if duplicate:
            decisions.append(duplicate)

        if not decisions:
            trusted = any(fnmatchcase(context.name, item) for item in self.config.trusted_tools)
            if trusted or not context.name.startswith("mcp_"):
                reason = "trusted tool default" if trusted else "local registered tool default"
                code = "trusted_tool.allow" if trusted else "local_tool.allow"
                decisions.append((PolicyAction.ALLOW, reason, code, None))
            else:
                decisions.append((PolicyAction(self.config.default_action), "unmatched tool default", "policy.default", None))

        rank = {PolicyAction.ALLOW: 0, PolicyAction.CONFIRM: 1, PolicyAction.DENY: 2}
        action = max((item[0] for item in decisions), key=rank.__getitem__)
        selected = [item for item in decisions if item[0] == action]
        return ToolPolicyDecision(
            action=action,
            reason="; ".join(dict.fromkeys(item[1] for item in selected)),
            reason_codes=list(dict.fromkeys(item[2] for item in selected)),
            rule_ids=list(dict.fromkeys(item[3] for item in selected if item[3])),
            argument_hash=signature,
        )

    def _matches(self, rule: ToolPolicyRuleConfig, context: ToolRequestContext) -> bool:
        if not any(fnmatchcase(context.name, pattern) for pattern in rule.tools):
            return False
        for argument, matcher in rule.arguments.items():
            value = context.arguments.get(argument)
            if value is None or not any(re.search(pattern, str(value)) for pattern in matcher.regex):
                return False
        return True

    def _path_decisions(
        self, context: ToolRequestContext
    ) -> list[tuple[PolicyAction, str, str, str | None]]:
        output: list[tuple[PolicyAction, str, str, str | None]] = []
        for argument in self.config.workspace.path_arguments.get(context.name, ()):
            raw = context.arguments.get(argument)
            if not isinstance(raw, str) or not raw.strip():
                continue
            parts = self._path_parts(raw)
            if ".." in parts:
                output.append((PolicyAction(self.config.workspace.traversal_action), "path traversal detected", "path.traversal", None))
            sensitive = {name.casefold() for name in self.config.workspace.sensitive_names}
            if any(part.casefold() in sensitive for part in parts):
                output.append((PolicyAction(self.config.workspace.sensitive_action), "sensitive path denied", "path.sensitive", None))
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = self.workspace / candidate
            resolved = candidate.resolve(strict=False)
            extra_allowed = (
                context.name in self.config.workspace.read_only_tools
                and any(self._is_under(resolved, root) for root in self.extra_read_roots)
            )
            if not self._is_under(resolved, self.workspace) and not extra_allowed:
                configured = (
                    self.config.workspace.restricted_outside_action
                    if (self.restrict_to_workspace or context.restrict_to_workspace)
                    else self.config.workspace.outside_action
                )
                output.append((PolicyAction(configured), "path is outside workspace", "path.outside_workspace", None))
        return output

    def _command_decisions(
        self, context: ToolRequestContext
    ) -> list[tuple[PolicyAction, str, str, str | None]]:
        command_argument = self.config.command_arguments.get(context.name)
        if command_argument is None:
            return []
        command = context.arguments.get(command_argument)
        if not isinstance(command, str):
            return []
        output = []
        if any(pattern.search(command) for pattern in self._DANGEROUS_COMMANDS):
            output.append((PolicyAction.DENY, "dangerous command pattern", "command.dangerous", None))
        # Avoid DNS during policy evaluation; literal private/internal targets are deterministic.
        for match in self._URL_RE.finditer(command):
            host = urlsplit(match.group(0)).hostname or ""
            if host.casefold() in {"localhost", "localhost.localdomain"} or self._private_literal(host):
                output.append((PolicyAction.DENY, "internal/private URL detected", "network.private_target", None))
                break
        return output

    async def _duplicate_decision(
        self, context: ToolRequestContext, signature: str
    ) -> tuple[PolicyAction, str, str, str | None] | None:
        window = self.config.duplicate_window_seconds
        if window <= 0:
            return None
        scope = context.session_key or context.run_id or "global"
        key = (scope, context.name, signature)
        now = self._clock()
        async with self._seen_lock:
            expired = [item for item, seen in self._seen.items() if now - seen >= window]
            for item in expired:
                self._seen.pop(item, None)
            previous = self._seen.get(key)
            self._seen[key] = now
        if previous is not None and now - previous < window:
            return (PolicyAction(self.config.duplicate_action), "duplicate tool call within policy window", "duplicate.recent", None)
        return None

    @staticmethod
    def _path_parts(raw: str) -> tuple[str, ...]:
        normalized = raw.replace("\\", "/")
        return tuple(part for part in PurePath(normalized).parts if part not in {"/", "\\"})

    @staticmethod
    def _is_under(path: Path, root: Path) -> bool:
        try:
            common = os.path.commonpath((str(path), str(root)))
            return os.path.normcase(common) == os.path.normcase(str(root))
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _private_literal(host: str) -> bool:
        import ipaddress

        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        return address.is_private or address.is_loopback or address.is_link_local or address.is_reserved

    def audit_record(
        self,
        context: ToolRequestContext,
        decision: ToolPolicyDecision,
        **fields: Any,
    ) -> dict[str, Any]:
        clean_arguments = self.redactor.redact(context.arguments)
        clean_fields = self.redactor.redact(fields)
        trace_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"litebot:{context.run_id}:{context.session_key}:{context.tool_call_id}",
        ))
        timestamp = datetime.now(UTC).isoformat()
        attempt = max(1, int(context.attempt))
        events: list[dict[str, Any]] = []

        def add(stage: str, data: dict[str, Any]) -> None:
            sequence = len(events) + 1
            event = {
                "schema_version": "litebot-tool-audit/v2",
                "event_id": str(uuid.uuid4()),
                "trace_id": trace_id,
                "run_id": context.run_id,
                "session_key": context.session_key,
                "case_id": context.case_id,
                "tool_call_id": context.tool_call_id,
                "attempt": attempt,
                "sequence": sequence,
                "timestamp": timestamp,
                "stage": stage,
                "tool": context.name,
                "data": self.redactor.redact(data),
                "previous_event_hash": events[-1]["event_hash"] if events else None,
                "redaction": {
                    "applied": True,
                    "version": "v2",
                    "replacement": self.redactor.replacement,
                },
            }
            canonical = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
            event["event_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
            events.append(event)

        add("request", {"tool_call_id": context.tool_call_id, "tool": context.name})
        add("policy_decision", {
            "action": decision.action.value,
            "reason": decision.reason,
            "reason_codes": decision.reason_codes,
            "rule_ids": decision.rule_ids,
            "confirmation": decision.confirmation,
        })
        add("arguments", {
            "arguments": clean_arguments,
            "argument_hash": decision.argument_hash,
            "normalized_arguments": fields.get("normalized_arguments"),
        })
        outcome = str(fields.get("outcome", ""))
        stage = str(fields.get("stage", ""))
        if outcome != "pending":
            denied = decision.action is PolicyAction.DENY or stage == "denied"
            declined = decision.confirmation in {"declined", "error"}
            if not denied and not declined:
                add("execution_start", {"stage": stage or "execution"})
                add("execution_result", {
                    "outcome": outcome or "success",
                    "result_summary": fields.get("result_summary"),
                    "progress": fields.get("progress"),
                    "error": fields.get("error"),
                    "duration_ms": fields.get("duration_ms"),
                })
            if fields.get("retry_scheduled"):
                add("retry_scheduled", {
                    "previous_attempt": attempt,
                    "reason": fields.get("retry_reason"),
                    "retry_class": fields.get("retry_class"),
                    "backoff_ms": fields.get("backoff_ms"),
                })
            final_status = (
                "denied" if denied else
                "declined" if declined else
                "partial" if outcome == "partial" else
                "success" if outcome == "success" else
                "exhausted" if outcome == "exhausted" else
                "failed"
            )
            add("final_status", {"status": final_status, "audit_degraded": False})

        return {
            "schema_version": "litebot-tool-audit/v2",
            "trace_id": trace_id,
            "run_id": context.run_id,
            "session_key": context.session_key,
            "case_id": context.case_id,
            "tool_call_id": context.tool_call_id,
            "tool": context.name,
            "arguments": clean_arguments,
            "argument_hash": decision.argument_hash,
            "policy_action": decision.action.value,
            "policy_reason": decision.reason,
            "reason_codes": decision.reason_codes,
            "rule_ids": decision.rule_ids,
            "tool_trace": events,
            **clean_fields,
        }


def error_info_dict(info: ToolErrorInfo | None) -> dict[str, Any] | None:
    """Serialize error information without exposing enum implementation details."""
    if info is None:
        return None
    data = asdict(info)
    data["retry_class"] = info.retry_class.value
    return data
