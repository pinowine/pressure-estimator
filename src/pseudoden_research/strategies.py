from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import hypot
from typing import Protocol

from .config import StrategyConfig
from .entities import Snake
from .geometry import SQRT_TWO, Vec2, path_distance
from .world import WorldState


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


class PathStrategy(Protocol):
    def plan(self, world: WorldState, snake: Snake, target: Vec2) -> PathDecision:
        ...


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
