from __future__ import annotations

from datetime import datetime
from math import cos, pi, sin
from pathlib import Path

import pandas as pd

from pseudoden_research.config import TelemetryConfig
from pseudoden_research.geometry import Vec2
from pseudoden_research.simulation import (
    GameSimulation,
    _comparison_strategy,
    _decision_move,
    _distance_band,
    _fixed_comparison_input,
    _teacher_move,
)
from pseudoden_research.strategies import AStarStrategy

STRATEGIES = ("astar", "ml")
DEFAULT_ROUTES = ("box", "zigzag", "orbit", "diagonal_sweep")
ROUTE_DESCRIPTIONS = {
    "box": "Long straight segments, useful as the basic comparison route",
    "zigzag": "Frequent direction changes, useful for immediate response and action-change checks",
    "orbit": "Smooth circular input, useful for continuous pursuit checks",
    "diagonal_sweep": "Wide diagonal movement, useful for mid-range pressure and obstacle-detour checks",
}

def latest_log(directory: str, pattern: str) -> Path | None:
    files = sorted(Path(directory).glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None

def read_csv_or_empty(path: Path | None) -> pd.DataFrame:
    return pd.DataFrame() if path is None else pd.read_csv(path)

def collect_frame_trace(
    data_dir: Path,
    route_name: str,
    episodes: int,
    frames: int,
    dt: float = 1.0 / 60.0,
) -> Path:
    # one trace per route keeps later plots easy to compare
    rows: list[dict[str, object]] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = data_dir / f"{route_name}_frame_trace_{stamp}.csv"
    for episode in range(episodes):
        for strategy_name in STRATEGIES:
            rows.extend(_collect_episode_trace(strategy_name, route_name, episode, frames, dt))
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def build_episode_summary(data_dir: Path, route_name: str, trace: pd.DataFrame) -> Path:
    output_path = data_dir / f"{route_name}_episode_summary.csv"
    # episode summaries are the main table for line charts
    grouped = trace.groupby(["route", "strategy", "episode"], as_index=False)
    summary = grouped.agg(
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
        mean_distance_improvement=("distance_improvement", "mean"),
    )
    summary.to_csv(output_path, index=False)
    return output_path

def build_experience_long(data_dir: Path, route_name: str, summary: pd.DataFrame, source_log: Path) -> Path:
    output_path = data_dir / f"{route_name}_experience_metrics_long.csv"
    overall_metrics = [
        "teacher_agreement_rate",
        "move_change_rate",
        "closing_rate",
        "mean_distance_improvement",
        "avg_path_nodes",
        "avg_path_distance",
    ]
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        for metric in overall_metrics:
            rows.append(_metric_row(source_log, row, "overall", metric, row.get(metric)))
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path

def build_distance_advantage(data_dir: Path, route_name: str, trace: pd.DataFrame, bin_size: int = 100) -> Path:
    output_path = data_dir / f"{route_name}_distance_advantage.csv"
    df = trace.copy()
    # distance bins show where ml feels better or worse
    df["distance_start"] = (df["distance"] // bin_size * bin_size).astype(int)
    grouped = (
        df.groupby(["distance_start", "strategy"], as_index=False)
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
    for metric in [
        "frames",
        "closing_rate",
        "response_score",
        "teacher_agreement",
        "move_change_rate",
        "mean_distance_improvement",
    ]:
        piece = grouped.pivot(index="distance_start", columns="strategy", values=metric)
        pieces.append(piece.add_prefix("").rename(columns={name: f"{name}_{metric}" for name in STRATEGIES}))
    wide = pd.concat(pieces, axis=1).reset_index().fillna(0.0)
    # keep only bins where both strategies actually appeared
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

def build_response_grid(data_dir: Path, route_name: str, trace: pd.DataFrame) -> Path:
    output_path = data_dir / f"{route_name}_response_grid.csv"
    cell_size = 40
    df = trace.copy()
    # grid response becomes the map heatmap input
    df["col"] = (df["snake_x"] // cell_size).astype(int)
    df["row"] = (df["snake_y"] // cell_size).astype(int)
    grid = (
        df.groupby(["strategy", "col", "row"], as_index=False)
        .agg(frames=("frame", "count"), mean_response_score=("response_score", "mean"))
        .sort_values(["strategy", "row", "col"])
    )
    grid.to_csv(output_path, index=False)
    return output_path

def route_description(route_name: str) -> str:
    return ROUTE_DESCRIPTIONS.get(route_name, "Custom route.")

def available_routes() -> tuple[str, ...]:
    return tuple(ROUTE_DESCRIPTIONS)

def _collect_episode_trace(
    strategy_name: str,
    route_name: str,
    episode: int,
    frames: int,
    dt: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    simulation = GameSimulation(
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
            # these values describe feel, not just success rate
            move_changed = None if move is None else int(previous_move is not None and move != previous_move)
            improvement = None if previous_distance is None else previous_distance - metrics.distance
            closing = None if improvement is None else int(improvement > 0)
            rows.append(
                {
                    "strategy": strategy_name,
                    "route": route_name,
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

def _route_input(route_name: str, frame: int, episode: int) -> Vec2:
    if route_name == "box":
        return _fixed_comparison_input(frame, episode)
    if route_name == "zigzag":
        phase = ((frame // 45) + episode) % 4
        patterns = (Vec2(1.0, 0.9), Vec2(1.0, -0.9), Vec2(-1.0, 0.9), Vec2(-1.0, -0.9))
        return patterns[phase]
    if route_name == "orbit":
        angle = frame / 55.0 + episode * pi / 6.0
        return Vec2(cos(angle), sin(angle))
    if route_name == "diagonal_sweep":
        phase = ((frame // 135) + episode) % 4
        patterns = (Vec2(1.0, 1.0), Vec2(-0.8, 1.0), Vec2(-1.0, -0.8), Vec2(0.8, -1.0))
        return patterns[phase]
    raise ValueError(f"Unknown route preset: {route_name}")

def _metric_row(source_log: Path, row: pd.Series, scope: str, metric: str, value: object) -> dict[str, object]:
    return {
        "source_log": str(source_log),
        "strategy": row["strategy"],
        "episode": row["episode"],
        "scope": scope,
        "metric": metric,
        "value": value,
    }

def _agreement(move: tuple[int, int] | None, teacher_move: tuple[int, int] | None) -> int | None:
    if move is None or teacher_move is None:
        return None
    return int(move == teacher_move)

def _response_score(move_changed: int | None, closing: int | None, agreement: int | None) -> float:
    # disagreement means style difference here, not a failure by itself
    move_part = 0.0 if move_changed is None else float(move_changed)
    closing_part = 0.0 if closing is None else float(closing)
    disagreement_part = 0.0 if agreement is None else 1.0 - float(agreement)
    return 0.45 * move_part + 0.35 * closing_part + 0.20 * disagreement_part
