import sys

from benchmarks.framework.cli import main as framework_main


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "ab-context":
        from benchmarks.ab_context import main as ab_main

        return ab_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "episodic-memory":
        from benchmarks.episodic_memory import main as episodic_main

        return episodic_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "hooks":
        from benchmarks.hooks import main as hooks_main

        return hooks_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "tool-safety":
        from benchmarks.tool_safety import main as tool_safety_main

        return tool_safety_main(sys.argv[2:])
    return framework_main()


if __name__ == "__main__":
    raise SystemExit(main())
