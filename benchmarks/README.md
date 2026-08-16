# LiteBot baseline benchmark

This directory is a benchmark harness, not a pytest suite. It exercises the
nanobot v0.1.4.post6 agent abstractions and writes versioned JSONL results for
future LiteBot regression comparisons.

## Suites

- `deterministic` (18 cases): fully offline, scripted provider and controlled
  fixture tools. This is the default suite.
- `integration` (4 cases): scripted provider with real filesystem, shell,
  local stdio MCP, and subagent components. Run explicitly.
- `live` (6 cases): configured real provider and selected real tools. It is
  opt-in and requires `--allow-live`.

Every case gets a disposable system-temporary workspace. Only copied artifacts
are retained under `benchmark_results/<run-id>/artifacts/`; normal nanobot
sessions, memory, and the repository workspace are not used as case workspaces.

## Commands

```powershell
python -m benchmarks
python -m benchmarks run --suite deterministic
python -m benchmarks run --suite integration
python -m benchmarks run --suite live --allow-live
python -m benchmarks run --suite live --allow-live --case live_web_search
python -m benchmarks compare --baseline benchmark_results/<baseline> --candidate benchmark_results/<candidate>
python -m benchmarks.live_context --output benchmark_results/live-context.json
python -m benchmarks.context_management --output benchmark_results/context-management.json
python -m benchmarks ab-context --suite long-context --repetitions 3
python -m benchmarks ab-context --suite large-tool-result --repetitions 3
python -m benchmarks episodic-memory --output benchmark_results/episodic-memory.json
python -m benchmarks hooks
python -m benchmarks hooks --case tool_run_order --repetitions 30
python -m benchmarks tool-safety
python -m benchmarks tool-safety --case duplicate_block
python -m benchmarks.tool_safety
```

`episodic-memory` is a fully offline deterministic benchmark for structured
history migration and lexical Top-K retrieval. It reports Recall@1/3, MRR,
duplicate suppression, injected characters, and p50/p95 retrieval latency.

`hooks` is a fully offline deterministic lifecycle regression suite. Its 14
cases verify session/run/LLM/tool ordering, payloads, failure isolation,
legacy `AgentHook` compatibility, state isolation, allow/deny/modify control,
and callback overhead. It uses scripted providers and fixed tools, never an
LLM judge or network service. Results contain exact traces plus correctness
rates and unexpected/missing/duplicate event counts.

`tool-safety` is a fixed, fully offline runtime-governance regression suite.
Its 13 cases cover ALLOW, explicit DENY, confirmation approval/rejection/error,
duplicate blocking and expiry, argument canonicalization, workspace JSONL audit
integrity/correlation and write degradation, invalid arguments, and explicit
partial execution outcomes. It uses a scripted provider, counting fake tools,
a fake approver, an injectable fake clock, and an injectable audit writer. Every
case gets an isolated temporary workspace; network and subprocess entry points
fail fast. Each result reports `case_name`, `task_success`, `audit_success`,
`strict_success`, and `failure_reason`. The command exits successfully only when
strict success is 100% for every selected case.

`ab-context` is the strict live-provider paired benchmark. It runs the same
`AgentLoop` in explicit `baseline` and `context_management` modes, records all
agent and rolling-summary model calls, and refuses to publish a reduction when
provider usage or paired control checks are incomplete. The long-context suite
is the primary overall result; large Tool Result measurements remain separate.

`benchmarks.live_context` runs five real-provider rolling-summary recall cases
using the provider selected in the local nanobot configuration. It never writes
provider credentials into benchmark results.

The live suite uses the normal nanobot configuration. Missing credentials,
network, MCP prerequisites, quotas, or external services are recorded as
`SKIPPED` where they can be identified as environment failures. Live results
use rule assertions only; there is no LLM judge.

## Output

Each run contains `results.jsonl`, `summary.json`, and optional case artifacts.
JSONL is flushed after every case. Tool arguments/results are truncated and
secret-like fields are redacted. Provider usage is accumulated per LLM call,
not taken from `AgentRunResult.usage`, which only represents its last response
in this nanobot version.

The context A/B trace separately records `get_tool_result` and
`search_tool_result` calls, artifact characters returned by those tools, and
artifact-retrieval characters carried into each later model prompt. Provider
token usage remains attributed to the main agent; retrieval tools do not make
independent LLM calls.

Comparison treats `PASSED -> FAILED` as a correctness regression. Metrics are
reported but do not have hard thresholds. `PASSED -> SKIPPED` is a coverage
warning rather than a correctness failure.
