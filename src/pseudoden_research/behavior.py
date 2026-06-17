from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, radians, sin
from random import Random

from .config import SnakeConfig
from .entities import Player, Snake
from .geometry import Vec2
from .world import WorldState


@dataclass(frozen=True)
class Personality:
    Se: int
    Si: int
    Ne: int
    Ni: int
    Te: int
    Ti: int
    Fe: int
    Fi: int


def map_value(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    t = (value - in_min) / (in_max - in_min)
    return out_min + t * (out_max - out_min)


def generate_personality(rng: Random) -> Personality:
    return Personality(
        Se=rng.randrange(1, 100),
        Si=rng.randrange(1, 100),
        Ne=rng.randrange(1, 100),
        Ni=rng.randrange(1, 100),
        Te=rng.randrange(1, 100),
        Ti=rng.randrange(1, 100),
        Fe=rng.randrange(1, 100),
        Fi=rng.randrange(1, 100),
    )


def config_from_personality(personality: Personality, cell_size: int) -> SnakeConfig:
    p = personality
    # map personality scores into snake behavior tuning
    return SnakeConfig(
        move_speed=map_value((p.Se + p.Te) / 2, 1, 100, 1, 5) * cell_size,
        turn_speed=map_value(p.Se, 1, 100, 0.05, 0.55),
        hearing_range=map_value(p.Fe, 1, 100, 10 * cell_size, 20 * cell_size),
        vision_range=map_value(p.Ni, 1, 100, 5 * cell_size, 10 * cell_size),
        vision_fov=map_value(p.Ti, 1, 100, radians(60), radians(120)),
        hearing_confidence_bias=map_value((p.Ti + p.Ni) / 2, 1, 100, 0.2, 0.8),
        max_chase_time=map_value(p.Si, 1, 100, 1.0, 6.0),
        lose_interest_time=map_value(p.Ne, 1, 100, 1.0, 4.0),
        distraction_chance=map_value((p.Ne + p.Fi) / 2, 1, 100, 0.01, 0.12),
        attack_cooldown=map_value((p.Te + p.Ti) / 2, 1, 100, 1.5, 0.4),
        attack_range=map_value(p.Se, 1, 100, 18, 40),
        body_segments=int(map_value(p.Si, 1, 100, 30, 60)),
        body_thickness=map_value(p.Fi, 1, 100, 10, 20),
        mass=map_value((p.Si + p.Fi) / 2, 1, 100, 1.0, 3.0),
        max_ceiling_time=map_value(p.Ni - p.Fi, -99, 99, 1.0, 5.0),
        can_ceiling_crawl=p.Se > 40,
        adhesion=map_value((p.Ni + p.Fe) / 2, 1, 100, 0.8, 1.5),
    )


def angle_normalize(angle: float) -> float:
    angle = (angle + pi) % (2 * pi)
    if angle <= 0:
        angle += 2 * pi
    return angle - pi


@dataclass
class SnakeSense:
    config: SnakeConfig
    last_heard_pos: Vec2 | None = None
    last_heard_strength: float = 0.0
    last_seen_pos: Vec2 | None = None
    see_timer: float = 999.0

    def update(self, snake: Snake, player: Player, dt: float, rng: Random) -> None:
        self._update_hearing(snake, player, dt, rng)
        self._update_vision(snake, player, dt)

    @property
    def alert_state(self) -> str:
        if self.last_seen_pos and self.see_timer < 0.2:
            return "seen"
        if self.last_heard_pos and self.last_heard_strength > 0.3:
            return "heard"
        return "idle"

    def _update_hearing(self, snake: Snake, player: Player, dt: float, rng: Random) -> None:
        dx = player.pos.x - snake.head.x
        dy = player.pos.y - snake.head.y
        distance = snake.head.distance_to(player.pos)

        if distance > self.config.hearing_range:
            self.last_heard_strength = max(0.0, self.last_heard_strength - dt)
            return

        strength = 1 - min(max(distance / self.config.hearing_range, 0.0), 1.0)
        strength = strength ** (1.0 / self.config.hearing_confidence_bias)

        base_angle = atan2(dy, dx)
        # weaker sound gives a noisier guessed position
        max_noise = map_value(1 - strength, 0, 1, radians(5), radians(60))
        estimated_angle = base_angle + rng.uniform(-max_noise, max_noise)
        estimated_distance = distance * rng.uniform(0.8, 1.2)

        self.last_heard_pos = Vec2(
            snake.head.x + estimated_distance * cos(estimated_angle),
            snake.head.y + estimated_distance * sin(estimated_angle),
        )
        self.last_heard_strength = strength

    def _update_vision(self, snake: Snake, player: Player, dt: float) -> None:
        dx = player.pos.x - snake.head.x
        dy = player.pos.y - snake.head.y
        distance = snake.head.distance_to(player.pos)

        if distance > self.config.vision_range:
            self.see_timer += dt
            return

        facing = snake.facing_dir
        angle_to_player = atan2(dy, dx)
        facing_angle = atan2(facing.y, facing.x)
        # vision only works inside the snake's forward cone
        angle_diff = angle_normalize(angle_to_player - facing_angle)

        if abs(angle_diff) > self.config.vision_fov / 2:
            self.see_timer += dt
            return

        self.last_seen_pos = player.pos.copy()
        self.see_timer = 0.0


@dataclass
class SnakeMind:
    config: SnakeConfig
    state: str = "IDLE"
    state_time: float = 0.0
    chase_time: float = 0.0
    attack_cooldown_timer: float = 0.0
    current_target: Vec2 | None = None

    def update(
        self,
        world: WorldState,
        snake: Snake,
        player: Player,
        sense: SnakeSense,
        dt: float,
        rng: Random,
    ) -> Vec2 | None:
        self.state_time += dt
        self.attack_cooldown_timer -= dt

        # small state machine for patrol, search, and chase behavior
        if self.state == "IDLE":
            self._update_idle(world, snake, sense, rng)
        elif self.state == "PATROL":
            self._update_patrol(world, snake, sense, rng)
        elif self.state == "SEARCH":
            self._update_search(world, snake, sense)
        elif self.state == "CHASE":
            self._update_chase(snake, player, sense, dt)
        elif self.state == "LOST":
            self._update_lost(snake, sense)

        return self.current_target

    def _update_idle(self, world: WorldState, snake: Snake, sense: SnakeSense, rng: Random) -> None:
        snake.set_speed_multiplier(0.2)
        if not self.current_target or self._near_target(snake, self.current_target, world):
            self.current_target = self._pick_patrol_point(world, snake, rng)
        if self.state_time > 2.0:
            self._transition_to("PATROL")
        self._check_for_player(sense)

    def _update_patrol(self, world: WorldState, snake: Snake, sense: SnakeSense, rng: Random) -> None:
        snake.set_speed_multiplier(0.5)
        if not self.current_target or self._near_target(snake, self.current_target, world):
            self.current_target = self._pick_patrol_point(world, snake, rng)
        self._check_for_player(sense)

    def _update_search(self, world: WorldState, snake: Snake, sense: SnakeSense) -> None:
        snake.set_speed_multiplier(0.8)
        if sense.last_seen_pos:
            self.current_target = sense.last_seen_pos.copy()
        elif sense.last_heard_pos:
            self.current_target = sense.last_heard_pos.copy()
        else:
            self._transition_to("PATROL")
            return

        if (
            self._near_target(snake, self.current_target, world)
            or sense.see_timer > self.config.lose_interest_time
        ):
            self._transition_to("LOST")

        if sense.last_seen_pos and sense.see_timer < 0.1:
            self._transition_to("CHASE")

    def _update_chase(self, snake: Snake, player: Player, sense: SnakeSense, dt: float) -> None:
        snake.set_speed_multiplier(1.2)
        self.chase_time += dt
        if sense.last_seen_pos:
            self.current_target = sense.last_seen_pos.copy()

        if snake.head.distance_to(player.pos) < self.config.attack_range and self.attack_cooldown_timer <= 0:
            self.attack_cooldown_timer = self.config.attack_cooldown

        if sense.see_timer > self.config.lose_interest_time:
            self._transition_to("LOST")
        if self.chase_time > self.config.max_chase_time:
            self._transition_to("IDLE")

    def _update_lost(self, snake: Snake, sense: SnakeSense) -> None:
        snake.set_speed_multiplier(0.0)
        if self.state_time > 2.0:
            self._transition_to("PATROL")
        self._check_for_player(sense)

    def _transition_to(self, state: str) -> None:
        self.state = state
        self.state_time = 0.0
        if state == "CHASE":
            self.chase_time = 0.0

    def _check_for_player(self, sense: SnakeSense) -> None:
        # seen beats heard when both signals are fresh
        if sense.last_seen_pos and sense.see_timer < 0.5:
            self._transition_to("CHASE")
            self.current_target = sense.last_seen_pos.copy()
            return
        if sense.last_heard_pos and sense.last_heard_strength > 0.5:
            self._transition_to("SEARCH")
            self.current_target = sense.last_heard_pos.copy()

    def _near_target(self, snake: Snake, target: Vec2 | None, world: WorldState) -> bool:
        if not target:
            return True
        return snake.head.distance_to(target) < world.cell_size * 0.8

    def _pick_patrol_point(self, world: WorldState, snake: Snake, rng: Random) -> Vec2:
        range_cells = 6
        base_col, base_row = world.world_to_cell(snake.head)
        for _ in range(16):
            cell = (
                base_col + rng.randint(-range_cells, range_cells),
                base_row + rng.randint(-range_cells, range_cells),
            )
            if world.is_walkable(cell):
                return world.cell_to_world(cell)
        return snake.head.copy()
