from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from random import Random

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
from .strategies import AStarStrategy, PathDecision, PathStrategy, make_strategy
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
    recompute_count: int
    snake_state: str
    alert_state: str


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
    fieldnames = ["episode", "frames", "avg_distance", "final_distance", "caught", "recomputes"]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for episode in range(episodes):
            simulation = GameSimulation(
                strategy=make_strategy("ml"),
                telemetry_config=TelemetryConfig(enabled=False),
                seed=7 + episode,
            )
            episode_distance = 0.0
            episode_caught = 0
            metrics: SimulationMetrics | None = None
            try:
                for frame in range(frames):
                    # headless
                    metrics = simulation.step(_scripted_player_input(simulation, frame, episode), dt)
                    episode_distance += metrics.distance
                    episode_caught += int(metrics.caught)
            finally:
                simulation.close()

            assert metrics is not None
            avg_distance = episode_distance / max(frames, 1)
            total_distance += episode_distance
            total_caught += episode_caught
            writer.writerow(
                {
                    "episode": episode,
                    "frames": frames,
                    "avg_distance": f"{avg_distance:.3f}",
                    "final_distance": f"{metrics.distance:.3f}",
                    "caught": episode_caught,
                    "recomputes": metrics.recompute_count,
                }
            )

    return {
        "episodes": episodes,
        "frames": frames,
        "avg_distance": total_distance / max(episodes * frames, 1),
        "caught": total_caught,
        "log": str(output_path),
    }


def _scripted_player_input(simulation: GameSimulation, frame: int, episode: int) -> Vec2:
    patterns = (Vec2(1.0, 0.2), Vec2(-0.4, 1.0), Vec2(-1.0, -0.2), Vec2(0.4, -1.0))
    pattern = patterns[((frame // 120) + episode) % len(patterns)]
    away = Vec2(
        simulation.player.pos.x - simulation.snake.head.x,
        simulation.player.pos.y - simulation.snake.head.y,
    ).normalized()
    # mix a repeatable route with a little escape behavior, so episodes are varied but stable
    return Vec2(pattern.x * 0.55 + away.x * 0.45, pattern.y * 0.55 + away.y * 0.45)


def _headless_log_path() -> Path:
    directory = Path("logs")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"headless_ml_train_{stamp}.csv"
