from __future__ import annotations
from dataclasses import dataclass, field
from math import ceil
from .config import WorldConfig
from .geometry import Vec2
from .maps import ObstacleMap, ObstacleRect, default_obstacle_map

@dataclass
class WorldState:
    config: WorldConfig
    obstacle_map: ObstacleMap = field(default_factory=default_obstacle_map)
    elapsed: float = 0.0

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def height(self) -> int:
        return self.config.height

    @property
    def cell_size(self) -> int:
        return self.config.cell_size

    @property
    def cols(self) -> int:
        return ceil(self.width / self.cell_size)

    @property
    def rows(self) -> int:
        return ceil(self.height / self.cell_size)

    def clamp_point(self, point: Vec2, radius: float = 0.0) -> None:
        point.x = min(max(point.x, radius), self.width - radius)
        point.y = min(max(point.y, radius), self.height - radius)

    def world_to_cell(self, point: Vec2) -> tuple[int, int]:
        col = int(point.x // self.cell_size)
        row = int(point.y // self.cell_size)
        col = min(max(col, 0), self.cols - 1)
        row = min(max(row, 0), self.rows - 1)
        return col, row

    def cell_to_world(self, cell: tuple[int, int]) -> Vec2:
        col, row = cell
        x = (col + 0.5) * self.cell_size
        y = (row + 0.5) * self.cell_size
        point = Vec2(x, y)
        self.clamp_point(point)
        return point

    def obstacle_rectangles(self) -> tuple[ObstacleRect, ...]:
        return self.obstacle_map.rectangles

    def point_is_walkable(self, point: Vec2) -> bool:
        return self.is_walkable(self.world_to_cell(point))

    def is_walkable(self, cell: tuple[int, int]) -> bool:
        col, row = cell
        return 0 <= col < self.cols and 0 <= row < self.rows and cell not in self.obstacle_map.blocked_cells

    def nearest_walkable_cell(self, cell: tuple[int, int]) -> tuple[int, int] | None:
        start = self._clamp_cell(cell)
        if self.is_walkable(start):
            return start

        max_radius = max(self.cols, self.rows)
        for radius in range(1, max_radius + 1):
            best_cell: tuple[int, int] | None = None
            best_distance = float("inf")
            for dc in range(-radius, radius + 1):
                for dr in range(-radius, radius + 1):
                    if abs(dc) != radius and abs(dr) != radius:
                        continue
                    candidate = (start[0] + dc, start[1] + dr)
                    if not self.is_walkable(candidate):
                        continue
                    distance = dc * dc + dr * dr
                    if distance < best_distance:
                        best_cell = candidate
                        best_distance = distance
            if best_cell:
                return best_cell
        return None

    def nearest_walkable_point(self, point: Vec2) -> Vec2:
        clamped = point.copy()
        self.clamp_point(clamped)
        if self.point_is_walkable(clamped):
            return clamped

        cell = self.nearest_walkable_cell(self.world_to_cell(clamped))
        if cell is None:
            return clamped
        return self.cell_to_world(cell)

    def _clamp_cell(self, cell: tuple[int, int]) -> tuple[int, int]:
        col, row = cell
        return min(max(col, 0), self.cols - 1), min(max(row, 0), self.rows - 1)
