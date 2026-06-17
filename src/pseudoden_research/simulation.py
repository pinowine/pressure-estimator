from __future__ import annotations

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
from .strategies import AStarStrategy, PathDecision, PathStrategy
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

    def reset_entities(self) -> None:
        center_y = self.world.height * 0.5
        center_x = self.world.width * 0.5
        self.player = Player(Vec2(center_x - 100.0, center_y))
        self.snake = Snake(Vec2(center_x + 100.0, center_y), config=self.snake_config)
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


def run_smoke_test(frames: int = 180, dt: float = 1.0 / 60.0) -> dict[str, float]:
    simulation = GameSimulation()
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
