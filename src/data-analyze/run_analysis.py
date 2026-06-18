from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
ROOT = SRC.parent
for path in (HERE, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnostics import create_diagnostic_figures
from plots import create_figures
from processor import (
    DEFAULT_ROUTES,
    available_routes,
    build_distance_advantage,
    build_episode_summary,
    build_experience_long,
    build_response_grid,
    collect_frame_trace,
    latest_log,
    read_csv_or_empty,
    route_description,
)

def run_visual_analysis(
    episodes: int = 20,
    frames: int = 600,
    routes: list[str] | None = None,
) -> dict[str, object]:
    # csv and figs stay outside src so only the analysis code is tracked
    data_dir = ROOT / "reports" / "data"
    figure_dir = ROOT / "reports" / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    training_log = latest_log(str(ROOT / "logs"), "headless_ml_train_*.csv")
    tuning_log = latest_log(str(ROOT / "logs"), "ml_tuning_*.csv")
    route_training_log = latest_log(str(ROOT / "logs"), "route_ml_training_*.csv")
    training = read_csv_or_empty(training_log)
    tuning = read_csv_or_empty(tuning_log)
    route_training = read_csv_or_empty(route_training_log)
    route_names = _normalize_routes(routes)

    route_results: list[dict[str, object]] = []
    traces: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for route_name in route_names:
        # each route has its own data and figure dirs
        route_data_dir = data_dir / route_name
        route_figure_dir = figure_dir / route_name
        route_data_dir.mkdir(parents=True, exist_ok=True)
        route_figure_dir.mkdir(parents=True, exist_ok=True)

        trace_path = collect_frame_trace(route_data_dir, route_name, max(1, episodes), max(1, frames))
        trace = read_csv_or_empty(trace_path)
        summary_path = build_episode_summary(route_data_dir, route_name, trace)
        summary = read_csv_or_empty(summary_path)
        traces.append(trace)
        summaries.append(summary)
        experience_path = build_experience_long(route_data_dir, route_name, summary, summary_path)
        distance_path = build_distance_advantage(route_data_dir, route_name, trace)
        response_grid_path = build_response_grid(route_data_dir, route_name, trace)
        distance_advantage = read_csv_or_empty(distance_path)
        response_grid = read_csv_or_empty(response_grid_path)
        figures = create_figures(
            route_figure_dir,
            route_name,
            route_description(route_name),
            summary,
            training,
            tuning,
            trace,
            distance_advantage,
            response_grid,
        )
        route_results.append(
            {
                "route": route_name,
                "description": route_description(route_name),
                "trace": str(trace_path),
                "data": [str(summary_path), str(experience_path), str(distance_path), str(response_grid_path)],
                "figures": [str(path) for path in figures],
                "metrics": _route_metrics(summary, distance_advantage),
            }
        )

    diagnostic_figures = create_diagnostic_figures(figure_dir / "_diagnostics", traces, summaries, route_training)

    return {
        "training_log": "" if training_log is None else str(training_log),
        "tuning_log": "" if tuning_log is None else str(tuning_log),
        "route_training_log": "" if route_training_log is None else str(route_training_log),
        "data_dir": str(data_dir),
        "figure_dir": str(figure_dir),
        "diagnostic_figures": [str(path) for path in diagnostic_figures],
        "routes": route_results,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Process and visualize A* vs ML experiment data.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument(
        "--routes",
        default=",".join(DEFAULT_ROUTES),
        help=f"Comma-separated route presets. Available: {', '.join(available_routes())}",
    )
    args = parser.parse_args()
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

def _normalize_routes(routes: list[str] | None) -> list[str]:
    route_names = list(DEFAULT_ROUTES if routes is None else routes)
    unknown = [route for route in route_names if route not in available_routes()]
    if unknown:
        raise ValueError(f"Unknown route presets: {', '.join(unknown)}")
    return route_names

def _route_metrics(summary, distance_advantage) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for strategy in ("astar", "ml"):
        rows = summary[summary["strategy"] == strategy]
        metrics[strategy] = {
            "avg_distance": float(rows["avg_distance"].mean()),
            "caught_total": float(rows["caught"].sum()),
            "teacher_agreement": float(rows["teacher_agreement_rate"].mean()),
            "move_change_rate": float(rows["move_change_rate"].mean()),
            "closing_rate": float(rows["closing_rate"].mean()),
            "path_distance": float(rows["avg_path_distance"].mean()),
        }
    metrics["best_response_bin"] = _best_row(distance_advantage, "response_advantage")
    metrics["best_closing_bin"] = _best_row(distance_advantage, "closing_advantage")
    return metrics

def _best_row(frame, metric: str) -> dict[str, float]:
    if frame.empty:
        return {}
    row = frame.loc[frame[metric].idxmax()]
    return {
        "distance_start": float(row["distance_start"]),
        "distance_end": float(row["distance_end"]),
        metric: float(row[metric]),
    }

if __name__ == "__main__":
    main()
