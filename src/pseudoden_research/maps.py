from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterator

Cell = tuple[int, int]

@dataclass(frozen=True)
class ObstacleRect:
    col: int
    row: int
    width: int
    height: int

    def iter_cells(self) -> Iterator[Cell]:
        for col in range(self.col, self.col + self.width):
            for row in range(self.row, self.row + self.height):
                yield col, row

@dataclass(frozen=True)
class ObstacleMap:
    key: str
    name: str
    test_goal: str
    rectangles: tuple[ObstacleRect, ...]

    @cached_property
    def blocked_cells(self) -> frozenset[Cell]:
        cells: set[Cell] = set()
        for rectangle in self.rectangles:
            cells.update(rectangle.iter_cells())
        return frozenset(cells)

def rect(col: int, row: int, width: int, height: int) -> ObstacleRect:
    return ObstacleRect(col, row, width, height)

# use grid coordinate to mark block on map
DEFAULT_OBSTACLE_MAP = ObstacleMap(
    key="sparse_blocks",
    name="Sparse Blocks",
    test_goal="short detours around isolated blockers",
    rectangles=(
        rect(10, 4, 2, 6),
        rect(18, 2, 6, 2),
        rect(19, 8, 2, 6),
        rect(28, 14, 6, 2),
        rect(6, 15, 5, 2),
        rect(31, 5, 2, 5),
    ),
)

def default_obstacle_map() -> ObstacleMap:
    return DEFAULT_OBSTACLE_MAP
