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
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Compare frozen ML against A* with the same scripted player route.",
    )
    parser.add_argument(
        "--visualize-analysis",
        action="store_true",
        help="Generate processed CSV data and figures for ML vs A* analysis.",
    )
    parser.add_argument(
        "--route-train-ml",
        action="store_true",
        help="Train the ML model on route-split A* teacher samples.",
    )
    parser.add_argument(
        "--map-test-analysis",
        action="store_true",
        help="Run A* vs ML held-out route tests across multiple obstacle maps.",
    )
    parser.add_argument("--episodes", type=int, default=12, help="Episodes for headless ML commands.")
    parser.add_argument("--frames", type=int, default=600, help="Frames per episode for headless ML commands.")
    parser.add_argument(
        "--routes",
        default="box,zigzag,orbit,diagonal_sweep",
        help="Comma-separated route presets for --visualize-analysis.",
    )
    parser.add_argument("--train-routes", default="box,zigzag,orbit", help="Routes used for route ML training.")
    parser.add_argument("--test-routes", default="diagonal_sweep", help="Held-out routes for route ML testing.")
    parser.add_argument("--route-passes", type=int, default=3, help="Partial-fit passes for route ML training.")
    parser.add_argument("--random-samples", type=int, default=512, help="Random map samples mixed into route training.")
    parser.add_argument(
        "--maps",
        default="sparse_blocks,corridor_gates,dense_blocks,narrow_passages",
        help="Comma-separated map presets for --map-test-analysis.",
    )
    parser.add_argument("--map-routes", default="diagonal_sweep", help="Held-out routes for --map-test-analysis.")
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

    if args.compare_strategies:
        from pseudoden_research.simulation import run_strategy_comparison

        summary = run_strategy_comparison(episodes=args.episodes, frames=args.frames)
        print(f"Strategy comparison complete: log={summary['log']}")
        for row in summary["strategies"]:  
            print(
                f"{row['strategy']}: "
                f"mean_avg_distance={row['mean_avg_distance']:.2f}, "
                f"caught_total={row['caught_total']}, "
                f"mean_recomputes={row['mean_recomputes']:.2f}, "
                f"mean_path_nodes={row['mean_path_nodes']:.2f}, "
                f"mean_path_distance={row['mean_path_distance']:.2f}, "
                f"teacher_agreement={row['mean_teacher_agreement']:.3f}, "
                f"move_change={row['mean_move_change_rate']:.3f}, "
                f"closing_rate={row['mean_closing_rate']:.3f}, "
                f"distance_improvement={row['mean_distance_improvement']:.3f}, "
                f"agreement_by_band={row['close_teacher_agreement']:.3f}/"
                f"{row['mid_teacher_agreement']:.3f}/{row['far_teacher_agreement']:.3f}, "
                f"closing_by_band={row['close_closing_rate']:.3f}/"
                f"{row['mid_closing_rate']:.3f}/{row['far_closing_rate']:.3f}, "
                f"move_by_band={row['close_move_change_rate']:.3f}/"
                f"{row['mid_move_change_rate']:.3f}/{row['far_move_change_rate']:.3f}"
            )
        return

    if args.visualize_analysis:
        analysis_dir = SRC / "data-analyze"
        if str(analysis_dir) not in sys.path:
            sys.path.insert(0, str(analysis_dir))
        from run_analysis import run_visual_analysis

        routes = [route.strip() for route in args.routes.split(",") if route.strip()]
        summary = run_visual_analysis(episodes=args.episodes, frames=args.frames, routes=routes)
        print(f"Visual analysis complete: data={summary['data_dir']}, figures={summary['figure_dir']}")
        for result in summary["routes"]:  
            print(f"Route: {result['route']}")  
            print(f"Frame trace: {result['trace']}")  
            for path in result["data"]:  
                print(f"Data: {path}")
            for path in result["figures"]:  
                print(f"Figure: {path}")
        for path in summary["diagnostic_figures"]:  
            print(f"Diagnostic figure: {path}")
        return

    if args.route_train_ml:
        analysis_dir = SRC / "data-analyze"
        if str(analysis_dir) not in sys.path:
            sys.path.insert(0, str(analysis_dir))
        from route_training import run_route_training

        summary = run_route_training(
            episodes=args.episodes,
            frames=args.frames,
            passes=args.route_passes,
            train_routes=tuple(route.strip() for route in args.train_routes.split(",") if route.strip()),
            test_routes=tuple(route.strip() for route in args.test_routes.split(",") if route.strip()),
            random_samples=args.random_samples,
        )
        print(f"Route ML training complete: log={summary['log']}")
        print(
            "Validation accuracy: "
            f"{summary['baseline_validation_accuracy']:.4f} -> {summary['best_validation_accuracy']:.4f}"
        )
        print(f"Test accuracy: {summary['baseline_test_accuracy']:.4f} -> {summary['best_test_accuracy']:.4f}")
        print(f"Deploy iteration: {summary['deploy_iteration']}")
        print(f"Promoted active model: {int(bool(summary['promoted']))}")
        print(f"Route best model: {summary['route_best_model']}")
        print(f"Route deploy model: {summary['route_deploy_model']}")
        return

    if args.map_test_analysis:
        analysis_dir = SRC / "data-analyze"
        if str(analysis_dir) not in sys.path:
            sys.path.insert(0, str(analysis_dir))
        from map_analysis import run_map_test_analysis

        summary = run_map_test_analysis(
            episodes=args.episodes,
            frames=args.frames,
            maps=tuple(item.strip() for item in args.maps.split(",") if item.strip()),
            routes=tuple(item.strip() for item in args.map_routes.split(",") if item.strip()),
        )
        print(f"Map test analysis complete: data={summary['data_dir']}, figures={summary['figure_dir']}")
        print(f"Overview data: {summary['overview_data']}")
        for path in summary["overview_figures"]:  
            print(f"Overview figure: {path}")
        for result in summary["maps"]:  
            print(f"Map: {result['map']}")  
            print(f"Trace: {result['trace']}")  
            for path in result["figures"]:  
                print(f"Figure: {path}")
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
