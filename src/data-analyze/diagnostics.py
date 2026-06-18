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
# concept ref: https://hrl.boyuai.com/chapter/3/%E6%A8%A1%E4%BB%BF%E5%AD%A6%E4%B9%A0/
# api refs: https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html
# api refs: https://seaborn.pydata.org/generated/seaborn.heatmap.html
# api refs: https://seaborn.pydata.org/generated/seaborn.lineplot.html
# api refs: https://seaborn.pydata.org/generated/seaborn.scatterplot.html
# api refs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

ACTION_ORDER = [
    "up-left",
    "up",
    "up-right",
    "left",
    "stay",
    "right",
    "down-left",
    "down",
    "down-right",
    "none",
]

def create_diagnostic_figures(
    figure_dir: Path,
    traces: list[pd.DataFrame],
    summaries: list[pd.DataFrame],
    route_training: pd.DataFrame,
) -> list[Path]:
    # these figs target imitation learning failure modes
    sns.set_theme(style="whitegrid", context="notebook")
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    figures = [
        plot_confusion_matrix(figure_dir / "05_imitation_confusion_matrix.png", trace),
        plot_compounding_error(figure_dir / "06_compounding_error_curve.png", trace),
        plot_occupancy_gap(figure_dir / "07_occupancy_gap_heatmap.png", trace),
        plot_experience_score(figure_dir / "08_experience_score_curve.png", summary),
        plot_data_efficiency(figure_dir / "09_data_efficiency_curve.png", route_training),
        plot_experience_pareto(figure_dir / "10_experience_pareto_scatter.png", summary),
    ]
    return figures

def plot_confusion_matrix(output_path: Path, trace: pd.DataFrame) -> Path:
    # source note: follows sklearn's true-vs-predicted confusion matrix layout
    ml = _ml_trace(trace)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    if ml.empty:
        _empty(ax, "No ML rows found")
    else:
        data = ml.copy()
        data["teacher_action"] = data.apply(lambda row: _action_label(row["teacher_dx"], row["teacher_dy"]), axis=1)
        data["model_action"] = data.apply(lambda row: _action_label(row["decision_dx"], row["decision_dy"]), axis=1)
        matrix = pd.crosstab(data["teacher_action"], data["model_action"], normalize="index")
        matrix = matrix.reindex(index=ACTION_ORDER, columns=ACTION_ORDER, fill_value=0.0)
        matrix = matrix.loc[(matrix.sum(axis=1) > 0), (matrix.sum(axis=0) > 0)]
        sns.heatmap(matrix, cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f", ax=ax)
        ax.set_xlabel("model action")
        ax.set_ylabel("teacher action")
        ax.set_title("ML action distribution by A* teacher action")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def plot_compounding_error(output_path: Path, trace: pd.DataFrame) -> Path:
    curve = _compounding_curve(trace)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    if curve.empty:
        _empty(ax, "No disagreement events found")
    else:
        sns.lineplot(data=curve, x="horizon", y="distance_delta", hue="route", marker="o", ax=ax)
        ax.axhline(0, color="#6b7280", lw=1, ls="--")
        ax.set_title("Distance drift after ML disagrees with A*")
        ax.set_xlabel("frames after disagreement")
        ax.set_ylabel("distance change")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def plot_occupancy_gap(output_path: Path, trace: pd.DataFrame) -> Path:
    # source note: adapts occupancy distribution comparison into a grid heatmap
    world = WorldState(WorldConfig())
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    if trace.empty:
        _empty(ax, "No trace rows found")
    else:
        data = trace.copy()
        data["col"] = (data["snake_x"] // world.cell_size).astype(int)
        data["row"] = (data["snake_y"] // world.cell_size).astype(int)
        grouped = data.groupby(["strategy", "row", "col"], as_index=False).size()
        grouped["density"] = grouped["size"] / grouped.groupby("strategy")["size"].transform("sum")
        pivot = grouped.pivot_table(index=["row", "col"], columns="strategy", values="density", fill_value=0.0)
        pivot["gap"] = pivot.get("ml", 0.0) - pivot.get("astar", 0.0)
        matrix = pivot["gap"].unstack("col").reindex(index=range(world.rows), columns=range(world.cols)).fillna(0.0)
        vmax = max(abs(float(matrix.min().min())), abs(float(matrix.max().max())), 0.001)
        sns.heatmap(matrix, cmap="vlag", center=0, vmin=-vmax, vmax=vmax, square=True, ax=ax)
        _draw_obstacles(ax, world)
        ax.set_title("Occupancy gap: ML density minus A* density")
        ax.set_xlabel("map column")
        ax.set_ylabel("map row")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def plot_experience_score(output_path: Path, summary: pd.DataFrame) -> Path:
    scored = _scored_summary(summary)
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    if scored.empty:
        _empty(ax, "No summary rows found")
    else:
        sns.lineplot(data=scored, x="episode", y="experience_score", hue="strategy", style="route", marker="o", ax=ax)
        ax.set_title("Episode-level experience score")
        ax.set_xlabel("episode")
        ax.set_ylabel("score")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def plot_data_efficiency(output_path: Path, route_training: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    if route_training.empty:
        _empty(ax, "No route training log found")
    else:
        data = route_training.copy()
        data["cumulative_samples"] = data["train_samples"] * (data["iteration"] + 1)
        long = data.melt(
            id_vars=["iteration", "cumulative_samples"],
            value_vars=["train_accuracy", "validation_accuracy", "test_accuracy", "random_accuracy"],
            var_name="metric",
            value_name="accuracy",
        )
        sns.lineplot(data=long, x="cumulative_samples", y="accuracy", hue="metric", marker="o", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title("Route training data-efficiency proxy")
        ax.set_xlabel("cumulative teacher samples seen")
        ax.set_ylabel("accuracy")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def plot_experience_pareto(output_path: Path, summary: pd.DataFrame) -> Path:
    # source note: uses seaborn scatterplot hue/style/size semantics
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    if summary.empty:
        _empty(ax, "No summary rows found")
    else:
        data = summary.copy()
        data["caught_size"] = data["caught"].clip(lower=1)
        sns.scatterplot(
            data=data,
            x="move_change_rate",
            y="closing_rate",
            hue="strategy",
            style="route",
            size="caught_size",
            sizes=(40, 240),
            alpha=0.78,
            ax=ax,
        )
        ax.set_title("Experience tradeoff: movement change vs closing")
        ax.set_xlabel("move change rate")
        ax.set_ylabel("closing rate")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def _ml_trace(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return trace
    cols = ["decision_dx", "decision_dy", "teacher_dx", "teacher_dy"]
    return trace[(trace["strategy"] == "ml")].dropna(subset=cols).copy()

def _compounding_curve(trace: pd.DataFrame) -> pd.DataFrame:
    ml = _ml_trace(trace)
    rows: list[dict[str, object]] = []
    horizons = (0, 15, 30, 60, 90, 120)
    for (route, episode), group in ml.groupby(["route", "episode"]):
        ordered = group.sort_values("frame").reset_index(drop=True)
        events = ordered[ordered["teacher_agreement"] == 0]
        for _, event in events.iterrows():
            base_distance = float(event["distance"])
            base_frame = int(event["frame"])
            for horizon in horizons:
                future = ordered[ordered["frame"] >= base_frame + horizon].head(1)
                if future.empty:
                    continue
                rows.append(
                    {
                        "route": route,
                        "episode": episode,
                        "horizon": horizon,
                        "distance_delta": float(future.iloc[0]["distance"]) - base_distance,
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby(["route", "horizon"], as_index=False)["distance_delta"].mean()

def _scored_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    data = summary.copy()
    data["experience_score"] = (
        data["closing_rate"] * 40.0
        + data["mean_distance_improvement"] * 4.0
        + data["caught"] * 2.0
        - data["move_change_rate"] * 8.0
        - data["avg_distance"] * 0.03
    )
    return data

def _action_label(dx: object, dy: object) -> str:
    if pd.isna(dx) or pd.isna(dy):
        return "none"
    move = (int(dx), int(dy))
    labels = {
        (-1, -1): "up-left",
        (0, -1): "up",
        (1, -1): "up-right",
        (-1, 0): "left",
        (0, 0): "stay",
        (1, 0): "right",
        (-1, 1): "down-left",
        (0, 1): "down",
        (1, 1): "down-right",
    }
    return labels.get(move, f"{move[0]},{move[1]}")

def _draw_obstacles(ax: plt.Axes, world: WorldState) -> None:
    for rect in world.obstacle_rectangles():
        ax.add_patch(
            Rectangle(
                (rect.col, rect.row),
                rect.width,
                rect.height,
                facecolor="#374151",
                edgecolor="#111827",
                linewidth=0.5,
            )
        )

def _empty(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.set_axis_off()
