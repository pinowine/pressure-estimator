from __future__ import annotations

import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from random import Random

from pseudoden_research.behavior import SnakeSense
from pseudoden_research.config import SnakeConfig, TelemetryConfig, WorldConfig
from pseudoden_research.entities import Player, Snake
from pseudoden_research.geometry import Vec2
from pseudoden_research.simulation import GameSimulation
from pseudoden_research.strategies import AStarStrategy
from pseudoden_research.world import WorldState


class ResearchCoreTests(unittest.TestCase):
    def test_astar_uses_diagonal_steps_in_open_world(self) -> None:
        world = WorldState(WorldConfig(width=200, height=200, cell_size=40))
        snake = Snake(Vec2(20, 20))
        decision = AStarStrategy().plan(world, snake, Vec2(180, 180))

        self.assertEqual(decision.cells[0], (0, 0))
        self.assertEqual(decision.cells[-1], (4, 4))
        self.assertLessEqual(len(decision.cells), 5)

    def test_player_moves_diagonally_and_stays_in_bounds(self) -> None:
        world = WorldState(WorldConfig(width=200, height=200, cell_size=40))
        player = Player(Vec2(20, 20))

        player.update(Vec2(-1, -1), 1.0, world)

        self.assertGreaterEqual(player.pos.x, player.radius)
        self.assertGreaterEqual(player.pos.y, player.radius)

    def test_snake_hears_player_outside_vision_fov(self) -> None:
        config = SnakeConfig(hearing_range=500, vision_range=500, vision_fov=1.0)
        snake = Snake(Vec2(200, 100), config=config)
        player = Player(Vec2(100, 100))
        sense = SnakeSense(config)

        snake.facing_dir.set(1, 0)
        sense.update(snake, player, 1.0 / 60.0, Random(1))

        self.assertEqual(sense.alert_state, "heard")
        self.assertIsNone(sense.last_seen_pos)
        self.assertIsNotNone(sense.last_heard_pos)

    def test_simulation_records_single_snake_metrics(self) -> None:
        simulation = GameSimulation(
            world=WorldState(WorldConfig(width=400, height=300, cell_size=40)),
            telemetry_config=TelemetryConfig(enabled=False),
        )
        try:
            metrics = simulation.step(Vec2(1, 0), 1.0 / 60.0)

            self.assertEqual(len(simulation.snake.segments), simulation.snake.config.body_segments)
            self.assertEqual(metrics.decision.algorithm, "A* baseline")
            self.assertGreaterEqual(metrics.decision.path_node_count, 1)
            self.assertEqual(metrics.alert_state, "seen")
            self.assertEqual(metrics.snake_state, "CHASE")
        finally:
            simulation.close()


if __name__ == "__main__":
    unittest.main()
