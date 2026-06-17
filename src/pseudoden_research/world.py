from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .config import WorldConfig
from .geometry import Vec2


@dataclass
class WorldState:
    config: WorldConfig
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

    def is_walkable(self, cell: tuple[int, int]) -> bool:
        col, row = cell
        return 0 <= col < self.cols and 0 <= row < self.rows
