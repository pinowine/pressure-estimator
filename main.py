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
    parser.add_argument(
        "--tune-ml",
        action="store_true",
        help="Run isolated ML tuning candidates and write a parameter result CSV.",
    )
    parser.add_argument("--episodes", type=int, default=12, help="Episodes for --train-ml or --tune-ml.")
    parser.add_argument("--frames", type=int, default=600, help="Frames per episode for --train-ml or --tune-ml.")
    parser.add_argument("--tune-limit", type=int, default=None, help="Limit how many tuning candidates to run.")
    parser.add_argument(
        "--analyze-ml-log",
        nargs="?",
        const="latest",
        default=None,
        help="Analyze a headless ML training CSV log. Omit the path to use the latest log.",
    )
    args = parser.parse_args()

    if args.analyze_ml_log is not None:
        from pseudoden_research.analysis import analyze_ml_training_log, format_ml_training_analysis

        summary = analyze_ml_training_log(None if args.analyze_ml_log == "latest" else args.analyze_ml_log)
        print(format_ml_training_analysis(summary))
        return

    if args.tune_ml:
        from pseudoden_research.tuning import format_ml_tuning_summary, run_ml_tuning

        summary = run_ml_tuning(episodes=args.episodes, frames=args.frames, limit=args.tune_limit)
        print(format_ml_tuning_summary(summary))
        return

    # train args
    if args.train_ml:
        from pseudoden_research.simulation import run_headless_ml_training

        summary = run_headless_ml_training(episodes=args.episodes, frames=args.frames)
        eval_accuracy = summary["eval_accuracy"]
        eval_text = f", eval_accuracy={eval_accuracy:.3f}" if isinstance(eval_accuracy, float) else ""
        print(
            "Headless ML training complete: "
            f"episodes={summary['episodes']}, "
            f"frames={summary['frames']}, "
            f"caught={summary['caught']}, "
            f"avg_distance={summary['avg_distance']:.2f}"
            f"{eval_text}, "
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
