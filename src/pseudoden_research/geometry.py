from __future__ import annotations

from dataclasses import dataclass
from math import hypot, sqrt


@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def copy(self) -> "Vec2":
        return Vec2(self.x, self.y)

    def set(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def add_scaled(self, other: "Vec2", scale: float) -> None:
        self.x += other.x * scale
        self.y += other.y * scale

    def length(self) -> float:
        return hypot(self.x, self.y)

    def distance_to(self, other: "Vec2") -> float:
        return hypot(self.x - other.x, self.y - other.y)

    def normalized(self) -> "Vec2":
        length = self.length()
        if length <= 0.000001:
            return Vec2()
        return Vec2(self.x / length, self.y / length)

    def clamp_length(self, max_length: float) -> None:
        length = self.length()
        if length <= max_length or length <= 0.000001:
            return
        scale = max_length / length
        self.x *= scale
        self.y *= scale

    def damp(self, amount: float) -> None:
        self.x *= amount
        self.y *= amount
        if abs(self.x) < 0.001:
            self.x = 0.0
        if abs(self.y) < 0.001:
            self.y = 0.0


def subtract(a: Vec2, b: Vec2) -> Vec2:
    return Vec2(a.x - b.x, a.y - b.y)


def path_distance(points: list[Vec2]) -> float:
    total = 0.0
    for left, right in zip(points, points[1:]):
        total += left.distance_to(right)
    return total


SQRT_TWO = sqrt(2.0)
