from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .baseline import BaselineRegistry
from .loader import CaseRegistry
from .models import Category, Profile
from .report import compare, load_results, write_report
from .runner import run_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LiteBot deterministic regression evaluation")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("validate")
    listing = sub.add_parser("list")
    listing.add_argument("--profile", choices=[item.value for item in Profile], default="core")
    listing.add_argument("--category", choices=[item.value for item in Category])
    run = sub.add_parser("run")
    run.add_argument("--profile", choices=[item.value for item in Profile], default="core")
    run.add_argument("--category", choices=[item.value for item in Category])
    run.add_argument("--case")
    run.add_argument("--output", type=Path)
    run.add_argument("--run-id")
    run.add_argument("--allow-live", action="store_true")
    run.add_argument("--allow-judge", action="store_true")
    baseline = sub.add_parser("baseline")
    baseline_sub = baseline.add_subparsers(dest="baseline_command")
    promote = baseline_sub.add_parser("promote")
    promote.add_argument("--run", type=Path, required=True)
    promote.add_argument("--profile", required=True)
    promote.add_argument("--name", required=True)
    default = baseline_sub.add_parser("set-default")
    default.add_argument("--profile", required=True)
    default.add_argument("--name", required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--baseline", required=True)
    comparison.add_argument("--current", type=Path, required=True)
    comparison.add_argument("--output", type=Path)
    comparison.add_argument("--strict-performance", action="store_true")
    audit = sub.add_parser("audit")
    audit_sub = audit.add_subparsers(dest="audit_command")
    show = audit_sub.add_parser("show")
    show.add_argument("--run", type=Path)
    show.add_argument("--run-id")
    show.add_argument("--case", required=True)
    return parser


def _cases(args: argparse.Namespace):
    cases = CaseRegistry().load(args.profile)
    if getattr(args, "category", None):
        cases = [case for case in cases if case.category.value == args.category]
    if getattr(args, "case", None):
        cases = [case for case in cases if case.case_id == args.case]
        if not cases:
            raise ValueError(f"unknown case: {args.case}")
    return cases


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        values = ["run", "--profile", "core"]
    args = _parser().parse_args(values)
    try:
        if args.command == "validate":
            counts = CaseRegistry().validate_core_inventory()
            print(json.dumps({"valid": True, "core_cases": 72, "categories": counts}, indent=2))
            return 0
        if args.command == "list":
            cases = _cases(args)
            for case in cases:
                print(f"{case.case_id:52} {case.profile.value}")
            print(f"Total: {len(cases)}")
            return 0
        if args.command == "run":
            if args.profile == "live" and not args.allow_live:
                raise ValueError("live profile requires --allow-live")
            if args.profile == "judge" and not args.allow_judge:
                raise ValueError("judge profile requires --allow-judge")
            cases = _cases(args)
            if not cases:
                raise ValueError(f"profile contains no cases: {args.profile}")
            output, results = asyncio.run(run_suite(cases, args.output, args.run_id))
            print(f"Results: {output}")
            return 1 if any(result.status != "passed" for result in results) else 0
        if args.command == "baseline":
            registry = BaselineRegistry()
            if args.baseline_command == "promote":
                print(registry.promote(args.run, args.profile, args.name))
                return 0
            if args.baseline_command == "set-default":
                registry.set_default(args.profile, args.name)
                return 0
            raise ValueError("baseline subcommand is required")
        if args.command == "compare":
            baseline_path = BaselineRegistry().resolve(args.baseline)
            old, old_meta = load_results(baseline_path)
            current, current_meta = load_results(args.current)
            report = compare(old, current, baseline_meta=old_meta, current_meta=current_meta, strict_performance=args.strict_performance)
            output = args.output or args.current / "comparison"
            write_report(report, output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print(f"Report: {output}")
            return 0 if report["gate"]["passed"] else 1
        if args.command == "audit" and args.audit_command == "show":
            run = args.run
            if run is None and args.run_id:
                from .runner import RESULTS_ROOT
                run = RESULTS_ROOT / args.run_id
            if run is None:
                raise ValueError("audit show requires --run or --run-id")
            rows, _ = load_results(run)
            row = next((item for item in rows if item["case_id"] == args.case), None)
            if row is None:
                raise ValueError(f"case not found in run: {args.case}")
            print(json.dumps(row.get("tool_trace", []), ensure_ascii=False, indent=2))
            return 0
        _parser().print_help()
        return 2
    except (ValueError, FileNotFoundError, FileExistsError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
