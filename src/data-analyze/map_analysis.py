from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
SRC = HERE.parent
ROOT = SRC.parent
for path in (HERE, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from processor import _agreement, _response_score, _route_input, read_csv_or_empty
from pseudoden_research.config import TelemetryConfig, WorldConfig
from pseudoden_research.maps import DEFAULT_OBSTACLE_MAP, ObstacleMap, rect
from pseudoden_research.simulation import (
    GameSimulation,
    _comparison_strategy,
    _decision_move,
    _distance_band,
    _teacher_move,
)
from pseudoden_research.strategies import AStarStrategy
from pseudoden_research.world import WorldState


# --- this part comes from ---
# api refs: https://seaborn.pydata.org/generated/seaborn.heatmap.html
# api refs: https://seaborn.pydata.org/generated/seaborn.lineplot.html
# api refs: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html

STRATEGIES = ("astar", "ml")
DEFAULT_MAPS = ("sparse_blocks", "corridor_gates", "dense_blocks", "narrow_passages")
DEFAULT_TEST_ROUTES = ("diagonal_sweep",)

@dataclass(frozen=True)
class MapCase:
    key: str
    title: str
    description: str
    obstacle_map: ObstacleMap

MAP_CASES = {
    "sparse_blocks": MapCase(
        key="sparse_blocks",
        title="Sparse Blocks",
        description="The original sparse obstacle map, useful as a baseline layout",
        obstacle_map=DEFAULT_OBSTACLE_MAP,
    ),
    "corridor_gates": MapCase(
        key="corridor_gates",
        title="Corridor Gates",
        description="Several wall segments with gates, useful for testing local bottlenecks",
        obstacle_map=ObstacleMap(
            key="corridor_gates",
            name="Corridor Gates",
            test_goal="detour through gates",
            rectangles=(
                rect(12, 0, 2, 8),
                rect(12, 13, 2, 9),
                rect(25, 0, 2, 9),
                rect(25, 14, 2, 8),
                rect(17, 5, 7, 1),
                rect(17, 16, 7, 1),
            ),
        ),
    ),
    "dense_blocks": MapCase(
        key="dense_blocks",
        title="Dense Blocks",
        description="Many isolated obstacles, useful for testing local avoidance and stability",
        obstacle_map=ObstacleMap(
            key="dense_blocks",
            name="Dense Blocks",
            test_goal="many local obstacle decisions",
            rectangles=(
                rect(5, 3, 3, 2),
                rect(10, 8, 3, 3),
                rect(15, 2, 4, 2),
                rect(16, 14, 3, 4),
                rect(23, 6, 2, 5),
                rect(27, 12, 4, 2),
                rect(32, 4, 3, 4),
                rect(34, 16, 3, 2),
            ),
        ),
    ),
    "narrow_passages": MapCase(
        key="narrow_passages",
        title="Narrow Passages",
        description="Long walls and narrow passages, useful for testing global planning pressure",
        obstacle_map=ObstacleMap(
            key="narrow_passages",
            name="Narrow Passages",
            test_goal="long walls with narrow passages",
            rectangles=(
                rect(8, 4, 12, 2),
                rect(22, 4, 11, 2),
                rect(8, 16, 12, 2),
                rect(22, 16, 11, 2),
                rect(18, 7, 2, 7),
                rect(27, 8, 2, 6),
            ),
        ),
    ),
}

def run_map_test_analysis(
    episodes: int = 20,
    frames: int = 600,
    maps: tuple[str, ...] = DEFAULT_MAPS,
    routes: tuple[str, ...] = DEFAULT_TEST_ROUTES,
) -> dict[str, object]:
    # maps are held out from training to check real gen
    map_cases = _selected_maps(maps)
    data_root = ROOT / "reports" / "map_tests" / "data"
    figure_root = ROOT / "reports" / "map_tests" / "figures"
    data_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    summaries: list[pd.DataFrame] = []
    for map_case in map_cases:
        map_data_dir = data_root / map_case.key
        map_figure_dir = figure_root / map_case.key
        map_data_dir.mkdir(parents=True, exist_ok=True)
        map_figure_dir.mkdir(parents=True, exist_ok=True)
        trace_path = collect_map_trace(map_data_dir, map_case, routes, episodes, frames)
        trace = read_csv_or_empty(trace_path)
        summary_path = build_map_summary(map_data_dir, map_case.key, trace)
        summary = read_csv_or_empty(summary_path)
        distance_path = build_map_distance_advantage(map_data_dir, map_case.key, trace)
        response_grid_path = build_map_response_grid(map_data_dir, map_case.key, trace)
        distance = read_csv_or_empty(distance_path)
        response_grid = read_csv_or_empty(response_grid_path)
        figures = create_map_figures(map_figure_dir, map_case, summary, trace, distance, response_grid)
        summaries.append(summary)
        results.append(
            {
                "map": map_case.key,
                "title": map_case.title,
                "description": map_case.description,
                "trace": str(trace_path),
                "data": [str(summary_path), str(distance_path), str(response_grid_path)],
                "figures": [str(path) for path in figures],
                "metrics": _map_metrics(summary, distance),
            }
        )

    overview = pd.concat(summaries, ignore_index=True)
    overview_path = data_root / "map_overview_summary.csv"
    overview.to_csv(overview_path, index=False)
    overview_figures = create_overview_figures(figure_root, overview)
    return {
        "overview_data": str(overview_path),
        "overview_figures": [str(path) for path in overview_figures],
        "data_dir": str(data_root),
        "figure_dir": str(figure_root),
        "maps": results,
    }

def collect_map_trace(
    data_dir: Path,
    map_case: MapCase,
    routes: tuple[str, ...],
    episodes: int,
    frames: int,
    dt: float = 1.0 / 60.0,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = data_dir / f"{map_case.key}_trace_{stamp}.csv"
    rows: list[dict[str, object]] = []
    for route_name in routes:
        for episode in range(max(1, episodes)):
            for strategy_name in STRATEGIES:
                rows.extend(_collect_map_episode(map_case, strategy_name, route_name, episode, max(1, frames), dt))
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path

def build_map_summary(data_dir: Path, map_key: str, trace: pd.DataFrame) -> Path:
    output_path = data_dir / f"{map_key}_episode_summary.csv"
    summary = (
        trace.groupby(["map", "route", "strategy", "episode"], as_index=False)
        .agg(
            frames=("frame", "count"),
            avg_distance=("distance", "mean"),
            final_distance=("distance", "last"),
            caught=("caught", "sum"),
            recomputes=("recomputes", "last"),
            avg_path_nodes=("path_nodes", "mean"),
            avg_path_distance=("path_distance", "mean"),
            teacher_agreement_rate=("teacher_agreement", "mean"),
            move_change_rate=("move_changed", "mean"),
            closing_rate=("closing", "mean"),
            response_score=("response_score", "mean"),
            mean_distance_improvement=("distance_improvement", "mean"),
        )
        .fillna(0.0)
    )
    summary.to_csv(output_path, index=False)
    return output_path

def build_map_distance_advantage(data_dir: Path, map_key: str, trace: pd.DataFrame, bin_size: int = 100) -> Path:
    output_path = data_dir / f"{map_key}_distance_advantage.csv"
    df = trace.copy()
    # same bins across maps keeps the comparison readable
    df["distance_start"] = (df["distance"] // bin_size * bin_size).astype(int)
    grouped = (
        df.groupby(["map", "route", "distance_start", "strategy"], as_index=False)
        .agg(
            frames=("frame", "count"),
            closing_rate=("closing", "mean"),
            response_score=("response_score", "mean"),
            teacher_agreement=("teacher_agreement", "mean"),
            move_change_rate=("move_changed", "mean"),
            mean_distance_improvement=("distance_improvement", "mean"),
        )
        .fillna(0.0)
    )
    pieces = []
    for metric in ("frames", "closing_rate", "response_score", "teacher_agreement", "move_change_rate", "mean_distance_improvement"):
        piece = grouped.pivot(index=["map", "route", "distance_start"], columns="strategy", values=metric)
        pieces.append(piece.rename(columns={name: f"{name}_{metric}" for name in STRATEGIES}))
    wide = pd.concat(pieces, axis=1).reset_index().fillna(0.0)
    wide = wide[(wide["astar_frames"] > 0) & (wide["ml_frames"] > 0)].copy()
    wide["distance_end"] = wide["distance_start"] + bin_size
    wide["distance_center"] = wide["distance_start"] + bin_size / 2
    wide["closing_advantage"] = wide["ml_closing_rate"] - wide["astar_closing_rate"]
    wide["response_advantage"] = wide["ml_response_score"] - wide["astar_response_score"]
    wide["agreement_gap"] = wide["ml_teacher_agreement"] - wide["astar_teacher_agreement"]
    wide["move_change_gap"] = wide["ml_move_change_rate"] - wide["astar_move_change_rate"]
    wide["improvement_advantage"] = wide["ml_mean_distance_improvement"] - wide["astar_mean_distance_improvement"]
    wide.to_csv(output_path, index=False)
    return output_path

def build_map_response_grid(data_dir: Path, map_key: str, trace: pd.DataFrame) -> Path:
    output_path = data_dir / f"{map_key}_response_grid.csv"
    cell_size = WorldConfig().cell_size
    df = trace.copy()
    df["col"] = (df["snake_x"] // cell_size).astype(int)
    df["row"] = (df["snake_y"] // cell_size).astype(int)
    grid = (
        df.groupby(["map", "route", "strategy", "col", "row"], as_index=False)
        .agg(frames=("frame", "count"), mean_response_score=("response_score", "mean"))
        .sort_values(["map", "route", "strategy", "row", "col"])
    )
    grid.to_csv(output_path, index=False)
    return output_path

def create_map_figures(
    figure_dir: Path,
    map_case: MapCase,
    summary: pd.DataFrame,
    trace: pd.DataFrame,
    distance: pd.DataFrame,
    response_grid: pd.DataFrame,
) -> list[Path]:
    # each map gets the same four views
    sns.set_theme(style="whitegrid", context="notebook")
    return [
        plot_map_response(figure_dir / "01_response_map.png", map_case, trace, response_grid),
        plot_map_experience(figure_dir / "02_experience_lines.png", map_case, summary),
        plot_map_distance_advantage(figure_dir / "03_distance_advantage.png", map_case, distance),
        plot_map_metric_bars(figure_dir / "04_metric_bars.png", map_case, summary),
    ]

def create_overview_figures(figure_dir: Path, overview: pd.DataFrame) -> list[Path]:
    # overview figs answer the cross-map question first
    sns.set_theme(style="whitegrid", context="notebook")
    figure_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_cross_map_summary(figure_dir / "00_cross_map_summary.png", overview),
        plot_cross_map_advantage(figure_dir / "00_cross_map_advantage.png", overview),
    ]

def plot_map_response(output_path: Path, map_case: MapCase, trace: pd.DataFrame, response_grid: pd.DataFrame) -> Path:
    world = WorldState(WorldConfig(), obstacle_map=map_case.obstacle_map)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    vmax = max(float(response_grid["mean_response_score"].max()), 0.01)
    route = _player_route(trace)
    for ax, strategy, cmap in zip(axes, STRATEGIES, ("Blues", "Reds")):
        grid = _response_matrix(response_grid, strategy, world.rows, world.cols)
        sns.heatmap(grid, ax=ax, cmap=cmap, vmin=0, vmax=vmax, cbar=True, square=True)
        _draw_obstacles(ax, world)
        if not route.empty:
            ax.plot(route["player_x"] / world.cell_size, route["player_y"] / world.cell_size, color="#111827", lw=1.6)
        ax.set_title(f"{strategy.upper()} response tendency")
        ax.set_xlabel("map column")
        ax.set_ylabel("map row")
    fig.suptitle(f"{map_case.key}: response heatmap")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def plot_map_experience(output_path: Path, map_case: MapCase, summary: pd.DataFrame) -> Path:
    metrics = {
        "teacher_agreement_rate": "A* agreement",
        "move_change_rate": "Move change",
        "closing_rate": "Closing rate",
        "avg_path_distance": "Path distance",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics.items()):
        sns.lineplot(data=summary, x="episode", y=metric, hue="strategy", style="route", marker="o", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.set_ylabel(metric)
    fig.suptitle(f"{map_case.key}: experience metrics")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def plot_map_distance_advantage(output_path: Path, map_case: MapCase, distance: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    long = distance.melt(
        id_vars=["route", "distance_center"],
        value_vars=["closing_advantage", "response_advantage", "move_change_gap", "improvement_advantage"],
        var_name="metric",
        value_name="value",
    )
    sns.lineplot(data=long, x="distance_center", y="value", hue="metric", style="route", marker="o", ax=ax)
    ax.axhline(0, color="#6b7280", lw=1, ls="--")
    ax.set_title(f"{map_case.key}: ML - A* by distance")
    ax.set_xlabel("distance")
    ax.set_ylabel("ML - A*")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def plot_map_metric_bars(output_path: Path, map_case: MapCase, summary: pd.DataFrame) -> Path:
    grouped = (
        summary.groupby(["route", "strategy"], as_index=False)
        .agg(
            avg_distance=("avg_distance", "mean"),
            caught=("caught", "sum"),
            closing_rate=("closing_rate", "mean"),
            move_change_rate=("move_change_rate", "mean"),
        )
    )
    long = grouped.melt(id_vars=["route", "strategy"], var_name="metric", value_name="value")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), ("avg_distance", "caught", "closing_rate", "move_change_rate")):
        sns.barplot(data=long[long["metric"] == metric], x="route", y="value", hue="strategy", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("route")
    fig.suptitle(f"{map_case.key}: aggregate metrics")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def plot_cross_map_summary(output_path: Path, overview: pd.DataFrame) -> Path:
    grouped = (
        overview.groupby(["map", "strategy"], as_index=False)
        .agg(avg_distance=("avg_distance", "mean"), caught=("caught", "sum"), closing_rate=("closing_rate", "mean"))
    )
    long = grouped.melt(id_vars=["map", "strategy"], var_name="metric", value_name="value")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, metric in zip(axes, ("avg_distance", "caught", "closing_rate")):
        sns.barplot(data=long[long["metric"] == metric], x="map", y="value", hue="strategy", ax=ax)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Cross-map A* vs ML summary")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def plot_cross_map_advantage(output_path: Path, overview: pd.DataFrame) -> Path:
    grouped = (
        overview.groupby(["map", "strategy"], as_index=False)
        .agg(
            avg_distance=("avg_distance", "mean"),
            closing_rate=("closing_rate", "mean"),
            move_change_rate=("move_change_rate", "mean"),
            response_score=("response_score", "mean"),
        )
    )
    pivot = grouped.pivot(index="map", columns="strategy")
    data = pd.DataFrame(
        {
            "map": pivot.index,
            "distance_advantage": pivot["avg_distance"]["astar"] - pivot["avg_distance"]["ml"],
            "closing_advantage": pivot["closing_rate"]["ml"] - pivot["closing_rate"]["astar"],
            "move_change_gap": pivot["move_change_rate"]["ml"] - pivot["move_change_rate"]["astar"],
            "response_advantage": pivot["response_score"]["ml"] - pivot["response_score"]["astar"],
        }
    )
    long = data.melt(id_vars=["map"], var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    sns.barplot(data=long, x="map", y="value", hue="metric", ax=ax)
    ax.axhline(0, color="#6b7280", lw=1, ls="--")
    ax.set_title("Cross-map ML advantage signals")
    ax.set_ylabel("advantage value")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def _collect_map_episode(
    map_case: MapCase,
    strategy_name: str,
    route_name: str,
    episode: int,
    frames: int,
    dt: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    world = WorldState(WorldConfig(), obstacle_map=map_case.obstacle_map)
    simulation = GameSimulation(
        world=world,
        strategy=_comparison_strategy(strategy_name),
        telemetry_config=TelemetryConfig(enabled=False),
        seed=7 + episode,
    )
    teacher = AStarStrategy()
    previous_move: tuple[int, int] | None = None
    previous_distance: float | None = None
    try:
        for frame in range(frames):
            metrics = simulation.step(_route_input(route_name, frame, episode), dt)
            move = _decision_move(metrics.decision)
            teacher_move = _teacher_move(simulation.world, metrics.decision, metrics.target, teacher)
            agreement = _agreement(move, teacher_move)
            # keep the same feel metrics used by route analysis
            move_changed = None if move is None else int(previous_move is not None and move != previous_move)
            improvement = None if previous_distance is None else previous_distance - metrics.distance
            closing = None if improvement is None else int(improvement > 0)
            rows.append(
                {
                    "map": map_case.key,
                    "route": route_name,
                    "strategy": strategy_name,
                    "episode": episode,
                    "frame": metrics.frame,
                    "player_x": metrics.player_pos.x,
                    "player_y": metrics.player_pos.y,
                    "snake_x": metrics.snake_pos.x,
                    "snake_y": metrics.snake_pos.y,
                    "distance": metrics.distance,
                    "caught": int(metrics.caught),
                    "recomputes": metrics.recompute_count,
                    "distance_band": _distance_band(metrics.distance),
                    "decision_dx": None if move is None else move[0],
                    "decision_dy": None if move is None else move[1],
                    "teacher_dx": None if teacher_move is None else teacher_move[0],
                    "teacher_dy": None if teacher_move is None else teacher_move[1],
                    "teacher_agreement": agreement,
                    "move_changed": move_changed,
                    "distance_improvement": improvement,
                    "closing": closing,
                    "path_nodes": metrics.decision.path_node_count,
                    "path_distance": metrics.decision.path_distance,
                    "response_score": _response_score(move_changed, closing, agreement),
                    "snake_state": metrics.snake_state,
                    "alert_state": metrics.alert_state,
                }
            )
            if move is not None:
                previous_move = move
            previous_distance = metrics.distance
    finally:
        simulation.close()
    return rows

def _map_metrics(summary: pd.DataFrame, distance: pd.DataFrame) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for strategy in STRATEGIES:
        rows = summary[summary["strategy"] == strategy]
        metrics[strategy] = {
            "avg_distance": float(rows["avg_distance"].mean()),
            "caught_total": float(rows["caught"].sum()),
            "teacher_agreement": float(rows["teacher_agreement_rate"].mean()),
            "move_change_rate": float(rows["move_change_rate"].mean()),
            "closing_rate": float(rows["closing_rate"].mean()),
            "response_score": float(rows["response_score"].mean()),
            "path_distance": float(rows["avg_path_distance"].mean()),
        }
    metrics["best_response_bin"] = _best_row(distance, "response_advantage")
    metrics["best_closing_bin"] = _best_row(distance, "closing_advantage")
    return metrics


def _response_matrix(response_grid: pd.DataFrame, strategy: str, rows: int, cols: int) -> pd.DataFrame:
    subset = response_grid[response_grid["strategy"] == strategy]
    matrix = subset.pivot(index="row", columns="col", values="mean_response_score")
    return matrix.reindex(index=range(rows), columns=range(cols)).fillna(0.0)


def _player_route(trace: pd.DataFrame) -> pd.DataFrame:
    route = trace[(trace["strategy"] == "astar") & (trace["episode"] == 0)].copy()
    return route.sort_values("frame")[["player_x", "player_y"]]

def _draw_obstacles(ax: plt.Axes, world: WorldState) -> None:
    for obstacle in world.obstacle_rectangles():
        ax.add_patch(
            Rectangle(
                (obstacle.col, obstacle.row),
                obstacle.width,
                obstacle.height,
                facecolor="#374151",
                edgecolor="#111827",
                linewidth=0.5,
            )
        )

def _best_row(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    if frame.empty:
        return {}
    row = frame.loc[frame[metric].idxmax()]
    return {
        "distance_start": float(row["distance_start"]),
        "distance_end": float(row["distance_end"]),
        metric: float(row[metric]),
    }

def _selected_maps(map_keys: tuple[str, ...]) -> list[MapCase]:
    unknown = [key for key in map_keys if key not in MAP_CASES]
    if unknown:
        raise ValueError(f"Unknown map cases: {', '.join(unknown)}")
    return [MAP_CASES[key] for key in map_keys]

def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())

def main() -> None:
    parser = argparse.ArgumentParser(description="Run held-out route tests across multiple obstacle maps.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--maps", default=",".join(DEFAULT_MAPS))
    parser.add_argument("--routes", default=",".join(DEFAULT_TEST_ROUTES))
    args = parser.parse_args()
    summary = run_map_test_analysis(
        episodes=args.episodes,
        frames=args.frames,
        maps=_split_csv(args.maps),
        routes=_split_csv(args.routes),
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

if __name__ == "__main__":
    main()
