from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import hypot
from pathlib import Path
from typing import Protocol

from .config import StrategyConfig
from .entities import Snake
from .geometry import SQRT_TWO, Vec2, path_distance
from .world import WorldState


MOVE_LABELS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


@dataclass
class PathDecision:
    algorithm: str
    cells: list[tuple[int, int]]
    points: list[Vec2]
    target: Vec2
    path_distance: float
    recomputed: bool = False

    @property
    def path_node_count(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class ModelTrainingReport:
    previous_model: bool
    train_samples: int
    eval_samples: int
    model_seen_steps_before: int
    model_seen_steps_after: int
    train_accuracy_before: float | None
    train_accuracy_after: float | None
    eval_accuracy_before: float | None
    eval_accuracy_after: float | None
    eval_prediction_changes: int | None
    eval_prediction_change_rate: float | None
    model_path: str


class PathStrategy(Protocol):
    def plan(self, world: WorldState, snake: Snake, target: Vec2) -> PathDecision:
        ...

# --- A* AGR START ---

@dataclass
class AStarStrategy:
    config: StrategyConfig = field(default_factory=StrategyConfig)
    last_goal_cell: tuple[int, int] | None = None
    last_plan_at: float = -999.0
    last_decision: PathDecision | None = None
    recompute_count: int = 0

    def plan(self, world: WorldState, snake: Snake, target: Vec2) -> PathDecision:
        start_cell = world.nearest_walkable_cell(world.world_to_cell(snake.head))
        raw_goal_cell = world.world_to_cell(target)
        goal_cell = world.nearest_walkable_cell(raw_goal_cell)
        if start_cell is None or goal_cell is None:
            return self._empty_decision(snake, recomputed=False)

        path_target = target.copy() if goal_cell == raw_goal_cell else world.cell_to_world(goal_cell)
        # reuse the last path until the target cell or timer changes
        should_recompute = (
            self.last_decision is None
            or goal_cell != self.last_goal_cell
            or world.elapsed - self.last_plan_at >= self.config.recompute_interval
        )

        if should_recompute:
            cells = self._find_path(world, start_cell, goal_cell)
            points = [world.cell_to_world(cell) for cell in cells]
            if points:
                points[-1] = path_target.copy()
            decision_target = path_target if points else snake.head.copy()
            self.last_decision = PathDecision(
                algorithm=self.config.algorithm_name,
                cells=cells,
                points=points,
                target=decision_target,
                path_distance=path_distance(points),
                recomputed=True,
            )
            self.last_goal_cell = goal_cell
            self.last_plan_at = world.elapsed
            self.recompute_count += 1
            return self.last_decision

        assert self.last_decision is not None
        self.last_decision.recomputed = False
        # cached decision keeps movement smooth between replans
        self.last_decision.target = path_target.copy()
        return self.last_decision

    def _empty_decision(self, snake: Snake, recomputed: bool) -> PathDecision:
        return PathDecision(
            algorithm=self.config.algorithm_name,
            cells=[],
            points=[],
            target=snake.head.copy(),
            path_distance=0.0,
            recomputed=recomputed,
        )

    def _find_path(
        self,
        world: WorldState,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        if not world.is_walkable(start) or not world.is_walkable(goal):
            return []

        # main path finding algorithm
        open_heap: list[tuple[float, int, tuple[int, int]]] = []
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        visited: set[tuple[int, int]] = set()
        serial = 0

        heappush(open_heap, (self._heuristic(start, goal), serial, start))

        while open_heap:
            _, _, current = heappop(open_heap)
            if current in visited:
                continue
            if current == goal:
                return self._reconstruct(came_from, current)

            visited.add(current)

            for neighbor, cost in self._neighbors(world, current):
                if neighbor in visited:
                    continue
                tentative = g_score[current] + cost
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                serial += 1
                priority = tentative + self._heuristic(neighbor, goal)
                heappush(open_heap, (priority, serial, neighbor))

        return []

    def _neighbors(
        self,
        world: WorldState,
        cell: tuple[int, int],
    ) -> list[tuple[tuple[int, int], float]]:
        col, row = cell
        result: list[tuple[tuple[int, int], float]] = []
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if dc == 0 and dr == 0:
                    continue
                neighbor = (col + dc, row + dr)
                if not world.is_walkable(neighbor):
                    continue
                if dc != 0 and dr != 0:
                    if not world.is_walkable((col + dc, row)) or not world.is_walkable((col, row + dr)):
                        continue
                # diagonal moves cost more than straight moves
                cost = SQRT_TWO if dc != 0 and dr != 0 else 1.0
                result.append((neighbor, cost))
        return result

    def _heuristic(self, left: tuple[int, int], right: tuple[int, int]) -> float:
        return hypot(left[0] - right[0], left[1] - right[1])

    def _reconstruct(
        self,
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

# --- MACHINE LEARNING START ---

@dataclass
class SklearnIncrementalStrategy:
    config: StrategyConfig = field(default_factory=lambda: StrategyConfig("ML sklearn SGD"))
    training_samples: int = 640
    evaluation_samples: int = 256
    training_offset: int = 0
    # the saved model lets later runs continue training instead of starting over
    model_path: Path = field(default_factory=lambda: Path("models") / "imitation_sgd.joblib")
    teacher: AStarStrategy = field(default_factory=AStarStrategy)
    model: object | None = None
    training_report: ModelTrainingReport | None = None
    recompute_count: int = 0

    def prepare_training(self, world: WorldState) -> ModelTrainingReport:
        self._fit(world)
        assert self.training_report is not None
        return self.training_report

    def plan(self, world: WorldState, snake: Snake, target: Vec2) -> PathDecision:
        # train lazily on the first planning call, so A* runs do not import sklearn
        self._fit(world)
        start = world.nearest_walkable_cell(world.world_to_cell(snake.head))
        raw_goal = world.world_to_cell(target)
        # if the live target is inside a wall, train/predict toward the nearest valid cell
        goal = world.nearest_walkable_cell(raw_goal)
        if start is None or goal is None:
            return PathDecision(self.config.algorithm_name, [], [], snake.head.copy(), 0.0, False)

        assert self.model is not None
        # the model returns a class id; MOVE_LABELS turns that id into a grid step
        label = int(self.model.predict([self._features(world, start, goal)])[0])  # type: ignore[union-attr]
        next_cell = self._choose_next_cell(world, start, goal, MOVE_LABELS[label])
        # this ML planner only commits to one step, then asks the model again next frame
        cells = [start] if next_cell == start else [start, next_cell]
        points = [world.cell_to_world(cell) for cell in cells]
        if next_cell == goal:
            points[-1] = target.copy() if goal == raw_goal else world.cell_to_world(goal)
        self.recompute_count += 1
        return PathDecision(self.config.algorithm_name, cells, points, points[-1].copy(), path_distance(points), True)

    def _fit(self, world: WorldState) -> None:
        # keep the trained model in memory after the first fit during this run
        if self.model is not None:
            return
        # joblib is the normal lightweight format for saving fitted models
        from joblib import dump, load
        from sklearn.linear_model import SGDClassifier

        features, labels = self._build_teacher_dataset(world, self.training_samples, self.training_offset)
        # fixed eval data makes different training iterations comparable
        eval_features, eval_labels = self._build_teacher_dataset(world, self.evaluation_samples, 10000)
        if not features:
            raise RuntimeError("ML training needs at least one reachable A* sample.")

        previous_model = self.model_path.exists()
        if previous_model:
            self.model = load(self.model_path)
        else:
            # SGDClassifier supports partial_fit
            self.model = SGDClassifier(loss="log_loss", random_state=0, learning_rate="optimal")

        train_before = self._score_model(self.model, features, labels) if previous_model else None
        eval_before = self._score_model(self.model, eval_features, eval_labels) if previous_model else None
        eval_predictions_before = self._predict_labels(self.model, eval_features) if previous_model else None
        seen_steps_before = int(getattr(self.model, "t_", 0) or 0)

        self.model.partial_fit(features, labels, classes=list(range(len(MOVE_LABELS))))  # type: ignore[attr-defined]
        seen_steps_after = int(getattr(self.model, "t_", 0) or 0)
        train_after = self._score_model(self.model, features, labels)
        eval_after = self._score_model(self.model, eval_features, eval_labels)
        eval_predictions_after = self._predict_labels(self.model, eval_features)
        eval_changes = self._count_prediction_changes(eval_predictions_before, eval_predictions_after)
        eval_change_rate = (
            eval_changes / len(eval_predictions_after)
            if eval_changes is not None and eval_predictions_after
            else None
        )

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(self.model, self.model_path)
        self.training_report = ModelTrainingReport(
            previous_model=previous_model,
            train_samples=len(features),
            eval_samples=len(eval_features),
            model_seen_steps_before=seen_steps_before,
            model_seen_steps_after=seen_steps_after,
            train_accuracy_before=train_before,
            train_accuracy_after=train_after,
            eval_accuracy_before=eval_before,
            eval_accuracy_after=eval_after,
            eval_prediction_changes=eval_changes,
            eval_prediction_change_rate=eval_change_rate,
            model_path=str(self.model_path),
        )

    def _build_teacher_dataset(
        self,
        world: WorldState,
        sample_count: int,
        offset: int,
    ) -> tuple[list[list[float]], list[int]]:
        # use only walkable cells so the teacher never labels impossible starts/goals
        cells = [(col, row) for col in range(world.cols) for row in range(world.rows) if world.is_walkable((col, row))]
        features: list[list[float]] = []
        labels: list[int] = []
        for index in range(min(sample_count, len(cells) * 4)):
            # deterministic sampling keeps experiments repeatable without storing a dataset yet
            sample_index = index + offset
            start = cells[(sample_index * 37) % len(cells)]
            goal = cells[(sample_index * 83 + 19) % len(cells)]
            # A* gives the full path; the model only learns the first step
            path = self.teacher._find_path(world, start, goal)
            if len(path) < 2:
                continue
            move = (path[1][0] - start[0], path[1][1] - start[1])
            if move in MOVE_LABELS:
                features.append(self._features(world, start, goal))
                labels.append(MOVE_LABELS.index(move))
        return features, labels

    def _score_model(self, model: object, features: list[list[float]], labels: list[int]) -> float | None:
        if not features:
            return None
        return float(model.score(features, labels))  # type: ignore[attr-defined]

    def _predict_labels(self, model: object, features: list[list[float]]) -> list[int]:
        if not features:
            return []
        return [int(label) for label in model.predict(features)]  # type: ignore[attr-defined]

    def _count_prediction_changes(self, before: list[int] | None, after: list[int]) -> int | None:
        if before is None:
            return None
        # prediction changes show whether this update actually moved the model
        return sum(1 for left, right in zip(before, after) if left != right)

    def _features(self, world: WorldState, start: tuple[int, int], goal: tuple[int, int]) -> list[float]:
        col, row = start
        # feature shape: 2 target offsets + 8 nearby walkability bits = 10 values
        # offsets are normalized so the model sees similar ranges on different map sizes
        values = [(goal[0] - col) / max(world.cols, 1), (goal[1] - row) / max(world.rows, 1)]
        values.extend(1.0 if world.is_walkable((col + dc, row + dr)) else 0.0 for dc, dr in MOVE_LABELS)
        return values

    def _choose_next_cell(
        self, world: WorldState, start: tuple[int, int], goal: tuple[int, int], predicted_move: tuple[int, int]
    ) -> tuple[int, int]:
        neighbors = [neighbor for neighbor, _ in self.teacher._neighbors(world, start)]
        preferred = (start[0] + predicted_move[0], start[1] + predicted_move[1])
        if preferred in neighbors:
            return preferred
        # if the model predicts a blocked move, fall back to the valid neighbor closest to goal
        return min(neighbors, key=lambda cell: self.teacher._heuristic(cell, goal), default=start)

# hook
def make_strategy(name: str) -> PathStrategy:
    if name == "ml":
        return SklearnIncrementalStrategy()
    if name == "astar":
        return AStarStrategy()
    raise ValueError(f"Unknown strategy: {name}")
