from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
# make local src importable when running the script directly
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="PseudoDen Python research runtime")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short headless simulation instead of opening the window.",
    )
    parser.add_argument(
        "--strategy",
        choices=("astar", "ml"),
        default="astar",
        help="Pass strategy to run.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        # quick headless check for CI and local sanity runs
        from pseudoden_research.simulation import run_smoke_test

        summary = run_smoke_test(strategy_name=args.strategy)
        print(
            "Smoke test complete: "
            f"frames={summary['frames']}, "
            f"recomputes={summary['recomputes']}, "
            f"distance={summary['distance']:.2f}"
        )
        return

    from pseudoden_research.app import run

    run(args.strategy)


if __name__ == "__main__":
    main()
