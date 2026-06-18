from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from pseudoden_research.config import WorldConfig
from pseudoden_research.world import WorldState


# --- this part comes from ---
# api refs: https://seaborn.pydata.org/generated/seaborn.heatmap.html
# api refs: https://seaborn.pydata.org/generated/seaborn.lineplot.html
# api refs: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
# the route figures are standard plotting adapters around project telemetry

METRICS = {
    "teacher_agreement_rate": "A* agreement",
    "move_change_rate": "Move change",
    "closing_rate": "Closing rate",
    "avg_path_distance": "Path distance",
}


def create_figures(
    figure_dir: Path,
    route_name: str,
    route_description: str,
    comparison: pd.DataFrame,
    training: pd.DataFrame,
    tuning: pd.DataFrame,
    trace: pd.DataFrame,
    distance_advantage: pd.DataFrame,
    response_grid: pd.DataFrame,
) -> list[Path]:
    # seaborn handles the chart styling, this file only shapes views
    sns.set_theme(style="whitegrid", context="notebook")
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        plot_response_map(figure_dir / "01_response_map.png", route_name, route_description, trace, response_grid),
        plot_experience_lines(figure_dir / "02_experience_lines.png", route_name, comparison),
        plot_training_optimization(figure_dir / "03_training_optimization.png", route_name, training, tuning),
        plot_distance_advantage(figure_dir / "04_distance_advantage.png", route_name, distance_advantage),
    ]
    return figures


def plot_response_map(
    output_path: Path,
    route_name: str,
    route_description: str,
    trace: pd.DataFrame,
    response_grid: pd.DataFrame,
) -> Path:
    world = WorldState(WorldConfig())
    # side-by-side heatmaps make the behavior diff visible fast
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    vmax = max(float(response_grid["mean_response_score"].max()), 0.01)
    route = _player_route(trace)
    for ax, strategy, cmap in zip(axes, ["astar", "ml"], ["Blues", "Reds"]):
        grid = _response_matrix(response_grid, strategy, world.rows, world.cols)
        sns.heatmap(grid, ax=ax, cmap=cmap, vmin=0, vmax=vmax, cbar=True, square=True)
        _draw_obstacles(ax, world)
        if not route.empty:
            ax.plot(route["player_x"] / world.cell_size, route["player_y"] / world.cell_size, color="#111827", lw=1.6)
        ax.set_title(f"{strategy.upper()} response tendency")
        ax.set_xlabel("map column")
        ax.set_ylabel("map row")
    fig.suptitle(f"{route_name}: map response heatmap")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def plot_experience_lines(output_path: Path, route_name: str, comparison: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), METRICS.items()):
        sns.lineplot(data=comparison, x="episode", y=metric, hue="strategy", marker="o", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.set_ylabel(metric)
    fig.suptitle(f"{route_name}: experience metrics by episode")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def plot_training_optimization(
    output_path: Path,
    route_name: str,
    training: pd.DataFrame,
    tuning: pd.DataFrame,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    if training.empty:
        axes[0].text(0.5, 0.5, "No training log found", ha="center", va="center")
    else:
        train_long = training.melt(
            id_vars=["iteration"],
            value_vars=["train_accuracy_after", "eval_accuracy_after", "best_eval_accuracy_after"],
            var_name="metric",
            value_name="value",
        ).dropna()
        sns.lineplot(data=train_long, x="iteration", y="value", hue="metric", marker="o", ax=axes[0])
    axes[0].set_title("Training accuracy")
    axes[0].set_ylim(0, 1)

    if tuning.empty:
        axes[1].text(0.5, 0.5, "No tuning log found", ha="center", va="center")
    else:
        tune_long = tuning.melt(
            id_vars=["candidate_index"],
            value_vars=["recent_eval_accuracy", "selection_score", "mean_prediction_change_rate"],
            var_name="metric",
            value_name="value",
        ).dropna()
        sns.lineplot(data=tune_long, x="candidate_index", y="value", hue="metric", marker="o", ax=axes[1])
    axes[1].set_title("Tuning candidates")
    axes[1].set_xlabel("candidate index")
    fig.suptitle(f"{route_name}: ML optimization history")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def plot_distance_advantage(output_path: Path, route_name: str, distance_advantage: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    # values above zero mean ml leads on that signal
    long = distance_advantage.melt(
        id_vars=["distance_center"],
        value_vars=["closing_advantage", "response_advantage", "move_change_gap", "improvement_advantage"],
        var_name="metric",
        value_name="value",
    )
    sns.lineplot(data=long, x="distance_center", y="value", hue="metric", marker="o", ax=ax)
    ax.axhline(0, color="#6b7280", lw=1, ls="--")
    ax.set_title(f"{route_name}: ML advantage by distance bin")
    ax.set_xlabel("distance")
    ax.set_ylabel("ML - A*")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def _response_matrix(response_grid: pd.DataFrame, strategy: str, rows: int, cols: int) -> pd.DataFrame:
    subset = response_grid[response_grid["strategy"] == strategy]
    matrix = subset.pivot(index="row", columns="col", values="mean_response_score")
    return matrix.reindex(index=range(rows), columns=range(cols)).fillna(0.0)

def _player_route(trace: pd.DataFrame) -> pd.DataFrame:
    route = trace[(trace["strategy"] == "astar") & (trace["episode"] == 0)].copy()
    return route.sort_values("frame")[["player_x", "player_y"]]

def _draw_obstacles(ax: plt.Axes, world: WorldState) -> None:
    for rect in world.obstacle_rectangles():
        patch = Rectangle(
            (rect.col, rect.row),
            rect.width,
            rect.height,
            facecolor="#374151",
            edgecolor="#111827",
            linewidth=0.5,
        )
        ax.add_patch(patch)
