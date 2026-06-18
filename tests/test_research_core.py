from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from random import Random

from pseudoden_research.analysis import analyze_ml_training_log
from pseudoden_research.behavior import SnakeSense
from pseudoden_research.config import SnakeConfig, TelemetryConfig, WorldConfig
from pseudoden_research.entities import Player, Snake
from pseudoden_research.geometry import Vec2
from pseudoden_research.maps import DEFAULT_OBSTACLE_MAP, ObstacleMap, ObstacleRect
from pseudoden_research.simulation import GameSimulation, run_headless_ml_training, run_strategy_comparison
from pseudoden_research.strategies import AStarStrategy
from pseudoden_research.tuning import run_ml_tuning
from pseudoden_research.world import WorldState


class ResearchCoreTests(unittest.TestCase):
    def test_astar_uses_diagonal_steps_in_open_world(self) -> None:
        world = WorldState(WorldConfig(width=200, height=200, cell_size=40))
        snake = Snake(Vec2(20, 20))
        decision = AStarStrategy().plan(world, snake, Vec2(180, 180))

        self.assertEqual(decision.cells[0], (0, 0))
        self.assertEqual(decision.cells[-1], (4, 4))
        self.assertLessEqual(len(decision.cells), 5)

    def test_astar_routes_around_obstacle_cells(self) -> None:
        obstacle_map = ObstacleMap(
            key="unit_wall",
            name="Unit Wall",
            test_goal="force a single lower gate",
            rectangles=(ObstacleRect(2, 0, 1, 4),),
        )
        world = WorldState(
            WorldConfig(width=200, height=200, cell_size=40),
            obstacle_map=obstacle_map,
        )
        snake = Snake(Vec2(20, 20))
        decision = AStarStrategy().plan(world, snake, Vec2(180, 20))

        self.assertTrue(decision.cells)
        self.assertTrue(all(cell not in obstacle_map.blocked_cells for cell in decision.cells))
        self.assertIn((2, 4), decision.cells)
        self.assertGreater(decision.path_distance, 160.0)

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

    def test_simulation_uses_single_obstacle_map_cleanly(self) -> None:
        simulation = GameSimulation(telemetry_config=TelemetryConfig(enabled=False))
        try:
            simulation.step(Vec2(1, 0), 1.0 / 60.0)

            self.assertEqual(simulation.current_map, DEFAULT_OBSTACLE_MAP)
            self.assertTrue(simulation.world.point_is_walkable(simulation.player.pos))
            self.assertTrue(simulation.world.point_is_walkable(simulation.snake.head))
        finally:
            simulation.close()

    def test_headless_ml_training_writes_model_evaluation(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                summary = run_headless_ml_training(episodes=1, frames=3)
                log_path = Path(str(summary["log"]))
                with log_path.open(newline="", encoding="utf-8") as file:
                    row = next(csv.DictReader(file))

                self.assertEqual(row["iteration"], "1")
                self.assertEqual(row["previous_model"], "0")
                self.assertEqual(row["feature_set"], "local_geometry_v2")
                self.assertEqual(row["feature_count"], "15")
                self.assertGreater(int(row["train_samples"]), 0)
                self.assertGreater(int(row["eval_samples"]), 0)
                self.assertNotEqual(row["eval_accuracy_after"], "")
                self.assertEqual(row["saved_best_model"], "1")
                self.assertTrue(Path(row["best_model_path"]).exists())
                self.assertTrue(Path("models/best_imitation_sgd.json").exists())

                analysis = analyze_ml_training_log(log_path)
                self.assertEqual(analysis["rows"], 1)
                self.assertEqual(analysis["saved_best_count"], 1)
                self.assertGreater(float(analysis["best_eval_accuracy"]), 0.0)
                # short smoke logs should not pretend the model is finished
                self.assertEqual(analysis["training_state"]["state"], "collecting_data")  # type: ignore[index]

                comparison = run_strategy_comparison(episodes=1, frames=3)
                comparison_log = Path(str(comparison["log"]))
                with comparison_log.open(newline="", encoding="utf-8") as file:
                    comparison_rows = list(csv.DictReader(file))
                    strategies = {row["strategy"] for row in comparison_rows}
                self.assertEqual(strategies, {"astar", "ml"})
                # The comparison log now includes experience-level behavior signals.
                self.assertIn("teacher_agreement_rate", comparison_rows[0])
                self.assertIn("move_change_rate", comparison_rows[0])
                self.assertIn("closing_rate", comparison_rows[0])
                self.assertIn("mean_distance_improvement", comparison_rows[0])
                self.assertIn("close_closing_rate", comparison_rows[0])
                self.assertIn("mid_move_change_rate", comparison_rows[0])
                self.assertIn("far_teacher_agreement_rate", comparison_rows[0])
            finally:
                os.chdir(original_cwd)

    def test_ml_tuning_writes_parameter_results(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                summary = run_ml_tuning(episodes=1, frames=3, limit=1)
                log_path = Path(str(summary["log"]))
                with log_path.open(newline="", encoding="utf-8") as file:
                    row = next(csv.DictReader(file))

                self.assertEqual(row["candidate"], "baseline_current")
                self.assertEqual(row["classifier_learning_rate"], "optimal")
                self.assertEqual(row["feature_set"], "local_geometry_v2")
                self.assertNotEqual(row["rationale"], "")
                self.assertNotEqual(row["selection_score"], "")
                self.assertTrue(Path(row["best_model_path"]).exists())
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
