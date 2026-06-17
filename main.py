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
    # mode args
    parser.add_argument(
        "--strategy",
        choices=("astar", "ml"),
        default="astar",
        help="Pass strategy to run.",
    )
    parser.add_argument(
        "--train-ml",
        action="store_true",
        help="Run headless ML training with scripted player control.",
    )
    parser.add_argument("--episodes", type=int, default=12, help="Training episodes for --train-ml.")
    parser.add_argument("--frames", type=int, default=600, help="Frames per episode for --train-ml.")
    args = parser.parse_args()

    # train args
    if args.train_ml:
        from pseudoden_research.simulation import run_headless_ml_training

        summary = run_headless_ml_training(episodes=args.episodes, frames=args.frames)
        print(
            "Headless ML training complete: "
            f"episodes={summary['episodes']}, "
            f"frames={summary['frames']}, "
            f"caught={summary['caught']}, "
            f"avg_distance={summary['avg_distance']:.2f}, "
            f"log={summary['log']}"
        )
        return

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
