from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import PlayerConfig, SnakeConfig
from .geometry import Vec2, subtract
from .world import WorldState


@dataclass
class Player:
    pos: Vec2
    config: PlayerConfig = field(default_factory=PlayerConfig)
    vel: Vec2 = field(default_factory=Vec2)
    eye_offset: Vec2 = field(default_factory=Vec2)

    @property
    def radius(self) -> float:
        return self.config.radius

    def update(self, input_dir: Vec2, dt: float, world: WorldState) -> None:
        previous = self.pos.copy()
        direction = input_dir.normalized()
        # acceleration based movement keeps diagonal input fair
        if direction.length() > 0:
            self.vel.add_scaled(direction, self.config.acceleration * dt)
            self.vel.clamp_length(self.config.max_speed)
        else:
            self.vel.damp(max(0.0, 1.0 - self.config.friction * dt))

        self.pos.add_scaled(self.vel, dt)
        self._keep_inside(world)
        if not world.point_is_walkable(self.pos):
            self.pos = previous
            self.vel.set(0.0, 0.0)
        self._update_eye()

    def _keep_inside(self, world: WorldState) -> None:
        before_x = self.pos.x
        before_y = self.pos.y
        world.clamp_point(self.pos, self.radius)
        if self.pos.x != before_x:
            self.vel.x = 0.0
        if self.pos.y != before_y:
            self.vel.y = 0.0

    def _update_eye(self) -> None:
        speed = self.vel.length()
        if speed <= 0.01:
            self.eye_offset.set(0.0, 0.0)
            return
        max_offset = self.radius * 0.7
        scale = min(speed / self.config.max_speed, 1.0) * max_offset / speed
        self.eye_offset.set(self.vel.x * scale, self.vel.y * scale)


@dataclass
class Snake:
    head: Vec2
    config: SnakeConfig = field(default_factory=SnakeConfig)
    vel: Vec2 = field(default_factory=Vec2)
    facing_dir: Vec2 = field(default_factory=lambda: Vec2(1.0, 0.0))
    speed_multiplier: float = 1.0
    segments: list[Vec2] = field(default_factory=list)
    path_points: list[Vec2] = field(default_factory=list)
    path_index: int = 0
    trail: deque[Vec2] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if not self.segments:
            self.segments = [self.head.copy() for _ in range(self.config.body_segments)]
        self.trail = deque(maxlen=self.config.trail_points)

    @property
    def speed(self) -> float:
        return self.config.move_speed * self.speed_multiplier

    @property
    def body_thickness(self) -> float:
        return self.config.body_thickness

    @property
    def capture_radius(self) -> float:
        return self.config.capture_radius

    def set_speed_multiplier(self, value: float) -> None:
        self.speed_multiplier = max(0.0, value)

    def set_path(self, points: list[Vec2]) -> None:
        self.path_points = [point.copy() for point in points]
        # skip the first point because it is usually the current cell
        self.path_index = 1 if len(self.path_points) > 1 else 0

    def update(self, dt: float, target: Vec2, world: WorldState) -> None:
        previous = self.head.copy()
        move_target = self._current_move_target(target)
        to_target = subtract(move_target, self.head)
        distance = to_target.length()

        if distance <= 0.001:
            self.vel.set(0.0, 0.0)
        else:
            direction = Vec2(to_target.x / distance, to_target.y / distance)
            step = min(self.speed * dt, distance)
            self.head.add_scaled(direction, step)
            self.vel.set(direction.x * self.speed, direction.y * self.speed)
            self.facing_dir.set(direction.x, direction.y)

        world.clamp_point(self.head, self.body_thickness)
        if not world.point_is_walkable(self.head):
            self.head = previous
            self.vel.set(0.0, 0.0)
        self.trail.append(self.head.copy())
        self._update_segments()

    def _current_move_target(self, target: Vec2) -> Vec2:
        while self.path_index < len(self.path_points):
            point = self.path_points[self.path_index]
            # advance through path points that are already reached
            if self.head.distance_to(point) > max(4.0, self.speed * 0.03):
                return point
            self.path_index += 1
        return target

    def _update_segments(self) -> None:
        if not self.segments:
            return
        self.segments[0].set(self.head.x, self.head.y)
        # each body segment chases the segment in front of it
        for index in range(1, len(self.segments)):
            prev = self.segments[index - 1]
            curr = self.segments[index]
            to_prev = subtract(prev, curr)
            distance = to_prev.length()
            if distance <= 0.001:
                continue
            extra = distance - self.config.segment_spacing
            if extra > 0:
                curr.add_scaled(Vec2(to_prev.x / distance, to_prev.y / distance), extra)
