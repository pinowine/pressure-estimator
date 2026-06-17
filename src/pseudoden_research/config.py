from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    width: int = 1600
    height: int = 880
    cell_size: int = 40
    background_color: tuple[int, int, int] = (13, 15, 18)


@dataclass(frozen=True)
class PlayerConfig:
    radius: float = 13.0
    acceleration: float = 1800.0
    max_speed: float = 330.0
    friction: float = 9.0


@dataclass(frozen=True)
class SnakeConfig:
    move_speed: float = 120.0
    turn_speed: float = 0.25
    hearing_range: float = 600.0
    vision_range: float = 300.0
    vision_fov: float = 1.6
    hearing_confidence_bias: float = 0.5
    max_chase_time: float = 3.5
    lose_interest_time: float = 2.5
    distraction_chance: float = 0.05
    attack_cooldown: float = 0.9
    attack_range: float = 28.0
    body_segments: int = 44
    body_thickness: float = 16.0
    mass: float = 2.0
    max_ceiling_time: float = 3.0
    can_ceiling_crawl: bool = True
    adhesion: float = 1.1
    trail_points: int = 90

    @property
    def segment_spacing(self) -> float:
        return self.body_thickness * 0.3

    @property
    def capture_radius(self) -> float:
        return self.attack_range


@dataclass(frozen=True)
class StrategyConfig:
    algorithm_name: str = "A* baseline"
    recompute_interval: float = 0.2


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = True
    interval: float = 0.1
    directory: str = "logs"
