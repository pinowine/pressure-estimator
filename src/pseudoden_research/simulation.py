from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from random import Random
from statistics import mean

from .behavior import (
    Personality,
    SnakeMind,
    SnakeSense,
    config_from_personality,
    generate_personality,
)
from .config import SnakeConfig, TelemetryConfig, WorldConfig
from .entities import Player, Snake
from .geometry import Vec2
from .maps import ObstacleMap
from .strategies import (
    AStarStrategy,
    ModelTrainingReport,
    PathDecision,
    PathStrategy,
    SklearnIncrementalStrategy,
    make_strategy,
)
from .telemetry import TelemetryWriter
from .world import WorldState


@dataclass
class SimulationMetrics:
    frame: int
    elapsed: float
    fps: float
    distance: float
    caught: bool
    decision: PathDecision
    player_pos: Vec2
    snake_pos: Vec2
    target: Vec2 | None
    recompute_count: int
    snake_state: str
    alert_state: str


@dataclass
class ExperienceStats:
    frames: int = 0
    teacher_matches: int = 0
    teacher_checks: int = 0
    move_changes: int = 0
    valid_moves: int = 0
    unique_moves: set[tuple[int, int]] = field(default_factory=set)
    closing_frames: int = 0
    distance_improvement_total: float = 0.0
    distance_improvement_count: int = 0
    previous_move: tuple[int, int] | None = None


EXPERIENCE_FIELDS = (
    "teacher_agreement_rate",
    "move_change_rate",
    "unique_move_count",
    "closing_rate",
    "mean_distance_improvement",
)
EXPERIENCE_BANDS = (("close", 300.0), ("mid", 600.0), ("far", float("inf")))


@dataclass
class GameSimulation:
    world: WorldState = field(default_factory=lambda: WorldState(WorldConfig()))
    strategy: PathStrategy = field(default_factory=AStarStrategy)
    telemetry_config: TelemetryConfig = field(default_factory=TelemetryConfig)
    seed: int = 7
    player: Player = field(init=False)
    snake: Snake = field(init=False)
    personality: Personality = field(init=False)
    snake_config: SnakeConfig = field(init=False)
    sense: SnakeSense = field(init=False)
    mind: SnakeMind = field(init=False)
    frame: int = 0
    last_metrics: SimulationMetrics | None = None
    telemetry: TelemetryWriter | None = field(default=None, init=False)
    rng: Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = Random(self.seed)
        # personality drives both sensing ranges and movement tuning
        self.personality = generate_personality(self.rng)
        self.snake_config = config_from_personality(self.personality, self.world.cell_size)
        self.sense = SnakeSense(self.snake_config)
        self.mind = SnakeMind(self.snake_config)
        self.telemetry = TelemetryWriter(
            directory=Path(self.telemetry_config.directory),
            interval=self.telemetry_config.interval,
            enabled=self.telemetry_config.enabled,
        )
        self.reset_entities()

    @property
    def current_map(self) -> ObstacleMap:
        return self.world.obstacle_map

    def reset_entities(self) -> None:
        center_y = self.world.height * 0.5
        center_x = self.world.width * 0.5
        player_pos = self.world.nearest_walkable_point(Vec2(center_x - 100.0, center_y))
        snake_pos = self.world.nearest_walkable_point(Vec2(center_x + 100.0, center_y))
        self.player = Player(player_pos)
        self.snake = Snake(snake_pos, config=self.snake_config)
        self.snake.facing_dir.set(-1.0, 0.0)
        self.sense = SnakeSense(self.snake_config)
        self.mind = SnakeMind(self.snake_config)
        if isinstance(self.strategy, AStarStrategy):
            self.strategy.last_decision = None
            self.strategy.last_goal_cell = None
            self.strategy.last_plan_at = -999.0

    def step(self, input_dir: Vec2, dt: float) -> SimulationMetrics:
        self.world.elapsed += dt
        self.frame += 1

        self.player.update(input_dir, dt, self.world)
        # sense first, then let the mind choose a target
        self.sense.update(self.snake, self.player, dt, self.rng)
        target = self.mind.update(self.world, self.snake, self.player, self.sense, dt, self.rng)

        if target:
            decision = self.strategy.plan(self.world, self.snake, target)
            if decision.recomputed:
                self.snake.set_path(decision.points)
            self.snake.update(dt, decision.target, self.world)
        else:
            # no target means hold position but still produce metrics
            algorithm = getattr(getattr(self.strategy, "config", None), "algorithm_name", "path strategy")
            decision = PathDecision(
                algorithm=algorithm,
                cells=[],
                points=[],
                target=self.snake.head.copy(),
                path_distance=0.0,
                recomputed=False,
            )
            self.snake.update(dt, self.snake.head, self.world)

        distance = self.snake.head.distance_to(self.player.pos)
        caught = distance <= self.snake.capture_radius + self.player.radius

        metrics = SimulationMetrics(
            frame=self.frame,
            elapsed=self.world.elapsed,
            fps=1.0 / dt if dt > 0 else 0.0,
            distance=distance,
            caught=caught,
            decision=decision,
            player_pos=self.player.pos.copy(),
            snake_pos=self.snake.head.copy(),
            target=target.copy() if target else None,
            recompute_count=getattr(self.strategy, "recompute_count", 0),
            snake_state=self.mind.state,
            alert_state=self.sense.alert_state,
        )
        self.last_metrics = metrics
        self._write_telemetry(metrics, dt, force=caught)

        if caught:
            # reset after capture so the next frame starts clean
            self.reset_entities()

        return metrics

    def close(self) -> None:
        if self.telemetry:
            self.telemetry.close()

    def _write_telemetry(self, metrics: SimulationMetrics, dt: float, force: bool) -> None:
        if not self.telemetry:
            return
        decision = metrics.decision
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame": metrics.frame,
            "algorithm": decision.algorithm,
            "player_x": f"{self.player.pos.x:.3f}",
            "player_y": f"{self.player.pos.y:.3f}",
            "snake_x": f"{self.snake.head.x:.3f}",
            "snake_y": f"{self.snake.head.y:.3f}",
            "distance": f"{metrics.distance:.3f}",
            "snake_speed": f"{self.snake.speed:.3f}",
            "target_x": f"{decision.target.x:.3f}",
            "target_y": f"{decision.target.y:.3f}",
            "path_nodes": decision.path_node_count,
            "path_distance": f"{decision.path_distance:.3f}",
            "caught": int(metrics.caught),
            "snake_state": metrics.snake_state,
            "alert_state": metrics.alert_state,
            "hearing_range": f"{self.snake_config.hearing_range:.3f}",
            "vision_range": f"{self.snake_config.vision_range:.3f}",
        }
        self.telemetry.write(row, dt, force=force)


def run_smoke_test(frames: int = 180, dt: float = 1.0 / 60.0, strategy_name: str = "astar") -> dict[str, float]:
    simulation = GameSimulation(strategy=make_strategy(strategy_name))
    try:
        for _ in range(frames):
            simulation.step(Vec2(0.55, 0.25), dt)
        metrics = simulation.last_metrics
        assert metrics is not None
        return {
            "frames": float(metrics.frame),
            "recomputes": float(metrics.recompute_count),
            "distance": metrics.distance,
        }
    finally:
        simulation.close()


def run_headless_ml_training(
    episodes: int = 12,
    frames: int = 600,
    dt: float = 1.0 / 60.0,
) -> dict[str, object]:
    episodes = max(1, episodes)
    frames = max(1, frames)
    output_path = _headless_log_path()
    total_distance = 0.0
    total_caught = 0
    last_training_report: ModelTrainingReport | None = None
    fieldnames = [
        "iteration",
        "episode",
        "frames",
        "feature_set",
        "feature_count",
        "previous_model",
        "train_samples",
        "eval_samples",
        "model_seen_steps_before",
        "model_seen_steps_after",
        "train_accuracy_before",
        "train_accuracy_after",
        "eval_accuracy_before",
        "eval_accuracy_after",
        "eval_accuracy_delta",
        "eval_prediction_changes",
        "eval_prediction_change_rate",
        "best_eval_accuracy_before",
        "best_eval_accuracy_after",
        "saved_best_model",
        "avg_distance",
        "final_distance",
        "caught",
        "recomputes",
        "model_path",
        "best_model_path",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for episode in range(episodes):
            strategy = SklearnIncrementalStrategy(training_offset=episode * 997)
            simulation = GameSimulation(
                strategy=strategy,
                telemetry_config=TelemetryConfig(enabled=False),
                seed=7 + episode,
            )
            episode_distance = 0.0
            episode_caught = 0
            metrics: SimulationMetrics | None = None
            try:
                training_report = strategy.prepare_training(simulation.world)
                last_training_report = training_report
                for frame in range(frames):
                    # headless means the script creates input instead of reading the keyboard
                    metrics = simulation.step(_scripted_player_input(simulation, frame, episode), dt)
                    episode_distance += metrics.distance
                    episode_caught += int(metrics.caught)
            finally:
                simulation.close()

            assert metrics is not None
            avg_distance = episode_distance / max(frames, 1)
            total_distance += episode_distance
            total_caught += episode_caught
            row = {
                "iteration": episode + 1,
                "episode": episode,
                "frames": frames,
                "avg_distance": f"{avg_distance:.3f}",
                "final_distance": f"{metrics.distance:.3f}",
                "caught": episode_caught,
                "recomputes": metrics.recompute_count,
            }
            row.update(_training_report_row(training_report))
            writer.writerow(row)

    return {
        "episodes": episodes,
        "frames": frames,
        "avg_distance": total_distance / max(episodes * frames, 1),
        "caught": total_caught,
        "eval_accuracy": None if last_training_report is None else last_training_report.eval_accuracy_after,
        "log": str(output_path),
    }


def run_strategy_comparison(
    episodes: int = 20,
    frames: int = 600,
    dt: float = 1.0 / 60.0,
) -> dict[str, object]:
    episodes = max(1, episodes)
    frames = max(1, frames)
    output_path = _comparison_log_path()
    rows: list[dict[str, object]] = []
    fieldnames = [
        "strategy",
        "episode",
        "frames",
        "avg_distance",
        "final_distance",
        "caught",
        "first_capture_frame",
        "recomputes",
        "avg_path_nodes",
        "avg_path_distance",
        *EXPERIENCE_FIELDS,
        *[
            f"{band}_{field}"
            for band, _ in EXPERIENCE_BANDS
            for field in ("frames", *EXPERIENCE_FIELDS)
        ],
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for episode in range(episodes):
            for strategy_name in ("astar", "ml"):
                row = _run_comparison_episode(strategy_name, episode, frames, dt)
                writer.writerow(row)
                rows.append(row)

    return {
        "episodes": episodes,
        "frames": frames,
        "log": str(output_path),
        "strategies": _comparison_summary(rows),
    }


def _run_comparison_episode(strategy_name: str, episode: int, frames: int, dt: float) -> dict[str, object]:
    simulation = GameSimulation(
        strategy=_comparison_strategy(strategy_name),
        telemetry_config=TelemetryConfig(enabled=False),
        seed=7 + episode,
    )
    distance_total = 0.0
    caught_total = 0
    first_capture_frame = ""
    path_nodes_total = 0.0
    path_distance_total = 0.0
    teacher = AStarStrategy()
    experience = ExperienceStats()
    band_experience = {band: ExperienceStats() for band, _ in EXPERIENCE_BANDS}
    previous_distance: float | None = None
    metrics: SimulationMetrics | None = None
    try:
        for frame in range(frames):
            metrics = simulation.step(_fixed_comparison_input(frame, episode), dt)
            distance_total += metrics.distance
            caught_total += int(metrics.caught)
            path_nodes_total += metrics.decision.path_node_count
            path_distance_total += metrics.decision.path_distance
            if metrics.caught and first_capture_frame == "":
                first_capture_frame = metrics.frame
            # these experience signals describe how the strategy feels, not raw CPU cost
            move = _decision_move(metrics.decision)
            teacher_move = _teacher_move(simulation.world, metrics.decision, metrics.target, teacher)
            improvement = None
            if previous_distance is not None:
                improvement = previous_distance - metrics.distance
            _record_experience(experience, move, teacher_move, improvement)
            _record_experience(band_experience[_distance_band(metrics.distance)], move, teacher_move, improvement)
            previous_distance = metrics.distance
    finally:
        simulation.close()

    assert metrics is not None
    row = {
        "strategy": strategy_name,
        "episode": episode,
        "frames": frames,
        "avg_distance": f"{distance_total / frames:.3f}",
        "final_distance": f"{metrics.distance:.3f}",
        "caught": caught_total,
        "first_capture_frame": first_capture_frame,
        "recomputes": metrics.recompute_count,
        "avg_path_nodes": f"{path_nodes_total / frames:.3f}",
        "avg_path_distance": f"{path_distance_total / frames:.3f}",
    }
    row.update(_experience_row(experience))
    for band, _ in EXPERIENCE_BANDS:
        row.update(_experience_row(band_experience[band], prefix=f"{band}_", include_frames=True))
    return row


def _comparison_strategy(name: str) -> PathStrategy:
    if name == "astar":
        return AStarStrategy()
    if name == "ml":
        return SklearnIncrementalStrategy(training_enabled=False)
    raise ValueError(f"Unknown comparison strategy: {name}")


def _comparison_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for strategy_name in ("astar", "ml"):
        strategy_rows = [row for row in rows if row["strategy"] == strategy_name]
        if not strategy_rows:
            continue
        summaries.append(
            {
                "strategy": strategy_name,
                "mean_avg_distance": mean(float(row["avg_distance"]) for row in strategy_rows),
                "caught_total": sum(int(row["caught"]) for row in strategy_rows),
                "mean_recomputes": mean(float(row["recomputes"]) for row in strategy_rows),
                "mean_path_nodes": mean(float(row["avg_path_nodes"]) for row in strategy_rows),
                "mean_path_distance": mean(float(row["avg_path_distance"]) for row in strategy_rows),
                "mean_teacher_agreement": mean(float(row["teacher_agreement_rate"]) for row in strategy_rows),
                "mean_move_change_rate": mean(float(row["move_change_rate"]) for row in strategy_rows),
                "mean_unique_move_count": mean(float(row["unique_move_count"]) for row in strategy_rows),
                "mean_closing_rate": mean(float(row["closing_rate"]) for row in strategy_rows),
                "mean_distance_improvement": mean(float(row["mean_distance_improvement"]) for row in strategy_rows),
                "close_teacher_agreement": _weighted_mean(strategy_rows, "close_teacher_agreement_rate", "close_frames"),
                "mid_teacher_agreement": _weighted_mean(strategy_rows, "mid_teacher_agreement_rate", "mid_frames"),
                "far_teacher_agreement": _weighted_mean(strategy_rows, "far_teacher_agreement_rate", "far_frames"),
                "close_closing_rate": _weighted_mean(strategy_rows, "close_closing_rate", "close_frames"),
                "mid_closing_rate": _weighted_mean(strategy_rows, "mid_closing_rate", "mid_frames"),
                "far_closing_rate": _weighted_mean(strategy_rows, "far_closing_rate", "far_frames"),
                "close_move_change_rate": _weighted_mean(strategy_rows, "close_move_change_rate", "close_frames"),
                "mid_move_change_rate": _weighted_mean(strategy_rows, "mid_move_change_rate", "mid_frames"),
                "far_move_change_rate": _weighted_mean(strategy_rows, "far_move_change_rate", "far_frames"),
            }
        )
    return summaries


def _record_experience(
    stats: ExperienceStats,
    move: tuple[int, int] | None,
    teacher_move: tuple[int, int] | None,
    distance_improvement: float | None,
) -> None:
    stats.frames += 1
    if move is not None:
        stats.unique_moves.add(move)
        if stats.previous_move is not None and move != stats.previous_move:
            stats.move_changes += 1
        stats.previous_move = move
        stats.valid_moves += 1
    if move is not None and teacher_move is not None:
        stats.teacher_checks += 1
        stats.teacher_matches += int(move == teacher_move)
    if distance_improvement is not None:
        stats.distance_improvement_total += distance_improvement
        stats.distance_improvement_count += 1
        if distance_improvement > 0:
            stats.closing_frames += 1


def _experience_row(
    stats: ExperienceStats,
    prefix: str = "",
    include_frames: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {}
    if include_frames:
        row[f"{prefix}frames"] = stats.frames
    row.update(
        {
            f"{prefix}teacher_agreement_rate": f"{_safe_rate(stats.teacher_matches, stats.teacher_checks):.3f}",
            f"{prefix}move_change_rate": f"{_safe_rate(stats.move_changes, max(stats.valid_moves - 1, 0)):.3f}",
            f"{prefix}unique_move_count": len(stats.unique_moves),
            f"{prefix}closing_rate": f"{_safe_rate(stats.closing_frames, stats.distance_improvement_count):.3f}",
            f"{prefix}mean_distance_improvement": (
                f"{_safe_mean(stats.distance_improvement_total, stats.distance_improvement_count):.3f}"
            ),
        }
    )
    return row


def _distance_band(distance: float) -> str:
    # fixed bands make repeated comparison logs easy to read together
    for band, upper_bound in EXPERIENCE_BANDS:
        if distance <= upper_bound:
            return band
    return "far"


def _weighted_mean(rows: list[dict[str, object]], value_key: str, weight_key: str) -> float:
    total_weight = sum(float(row[weight_key]) for row in rows)
    if total_weight <= 0:
        return 0.0
    return sum(float(row[value_key]) * float(row[weight_key]) for row in rows) / total_weight


def _decision_move(decision: PathDecision) -> tuple[int, int] | None:
    # the first grid step is the visible intention for this frame
    if len(decision.cells) < 2:
        return None
    start, next_cell = decision.cells[0], decision.cells[1]
    return next_cell[0] - start[0], next_cell[1] - start[1]


def _teacher_move(
    world: WorldState,
    decision: PathDecision,
    target: Vec2 | None,
    teacher: AStarStrategy,
) -> tuple[int, int] | None:
    # A* is used here as a stable teacher, so accuracy means: "same first choice"
    if target is None or not decision.cells:
        return None
    start = decision.cells[0]
    goal = world.nearest_walkable_cell(world.world_to_cell(target))
    if goal is None:
        return None
    path = teacher._find_path(world, start, goal)
    if len(path) < 2:
        return None
    return path[1][0] - start[0], path[1][1] - start[1]


def _safe_rate(count: int, total: int) -> float:
    # short smoke tests can have no valid denominator yet
    return count / total if total > 0 else 0.0


def _safe_mean(total: float, count: int) -> float:
    return total / count if count > 0 else 0.0


# evaluation
def _training_report_row(report: ModelTrainingReport) -> dict[str, object]:
    eval_delta = _optional_delta(report.eval_accuracy_after, report.eval_accuracy_before)
    return {
        "previous_model": int(report.previous_model),
        "feature_set": report.feature_set,
        "feature_count": report.feature_count,
        "train_samples": report.train_samples,
        "eval_samples": report.eval_samples,
        "model_seen_steps_before": report.model_seen_steps_before,
        "model_seen_steps_after": report.model_seen_steps_after,
        "train_accuracy_before": _format_optional_float(report.train_accuracy_before),
        "train_accuracy_after": _format_optional_float(report.train_accuracy_after),
        "eval_accuracy_before": _format_optional_float(report.eval_accuracy_before),
        "eval_accuracy_after": _format_optional_float(report.eval_accuracy_after),
        "eval_accuracy_delta": _format_optional_float(eval_delta),
        "eval_prediction_changes": "" if report.eval_prediction_changes is None else report.eval_prediction_changes,
        "eval_prediction_change_rate": _format_optional_float(report.eval_prediction_change_rate),
        "best_eval_accuracy_before": _format_optional_float(report.best_eval_accuracy_before),
        "best_eval_accuracy_after": _format_optional_float(report.best_eval_accuracy_after),
        "saved_best_model": int(report.saved_best_model),
        "model_path": report.model_path,
        "best_model_path": report.best_model_path,
    }


def _optional_delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return after - before


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def _scripted_player_input(simulation: GameSimulation, frame: int, episode: int) -> Vec2:
    patterns = (Vec2(1.0, 0.2), Vec2(-0.4, 1.0), Vec2(-1.0, -0.2), Vec2(0.4, -1.0))
    pattern = patterns[((frame // 120) + episode) % len(patterns)]
    away = Vec2(
        simulation.player.pos.x - simulation.snake.head.x,
        simulation.player.pos.y - simulation.snake.head.y,
    ).normalized()
    # mix a repeatable route with a little escape behavior, so episodes are varied but stable
    return Vec2(pattern.x * 0.55 + away.x * 0.45, pattern.y * 0.55 + away.y * 0.45)


def _fixed_comparison_input(frame: int, episode: int) -> Vec2:
    patterns = (Vec2(1.0, 0.2), Vec2(-0.4, 1.0), Vec2(-1.0, -0.2), Vec2(0.4, -1.0))
    # fixed input keeps both strategies facing the same player route
    return patterns[((frame // 120) + episode) % len(patterns)]


def _headless_log_path() -> Path:
    directory = Path("logs")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"headless_ml_train_{stamp}.csv"


def _comparison_log_path() -> Path:
    directory = Path("logs")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"strategy_comparison_{stamp}.csv"
