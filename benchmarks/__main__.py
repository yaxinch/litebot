import sys

from benchmarks.run import main as standard_main


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "ab-context":
        from benchmarks.ab_context import main as ab_main
        return ab_main(sys.argv[2:])
    return standard_main()

if __name__ == "__main__":
    raise SystemExit(main())
