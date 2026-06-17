from __future__ import annotations

import pyglet
from pyglet.window import key

from .config import TelemetryConfig
from .geometry import Vec2
from .simulation import GameSimulation


MIN_WINDOW_WIDTH = 900
MIN_WINDOW_HEIGHT = 520


class ResearchWindow(pyglet.window.Window):
    def __init__(self) -> None:
        super().__init__(
            width=1440,
            height=810,
            caption="PseudoDen Research - A* Baseline",
            resizable=True,
            vsync=True,
        )
        self.set_minimum_size(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        self.simulation = GameSimulation(
            telemetry_config=TelemetryConfig(enabled=True, directory="logs", interval=0.1)
        )
        self._arena_offset = Vec2()
        self._arena_scale = 1.0
        pyglet.clock.schedule_interval(self.update, 1.0 / 60.0)

    def update(self, dt: float) -> None:
        # cap large frame gaps so the simulation stays stable
        self.simulation.step(self._read_input(), min(dt, 1.0 / 20.0))

    def on_draw(self) -> None:
        self.clear()
        self._layout_arena()
        self._draw_background()
        self._draw_entities()

    def on_close(self) -> None:
        self.simulation.close()
        super().on_close()

    def _read_input(self) -> Vec2:
        left = self.keys[key.A] or self.keys[key.LEFT]
        right = self.keys[key.D] or self.keys[key.RIGHT]
        up = self.keys[key.W] or self.keys[key.UP]
        down = self.keys[key.S] or self.keys[key.DOWN]
        return Vec2(float(right) - float(left), float(down) - float(up))

    def _layout_arena(self) -> None:
        world = self.simulation.world
        scale = min(self.width / world.width, self.height / world.height)
        draw_width = world.width * scale
        draw_height = world.height * scale
        self._arena_scale = scale
        self._arena_offset.set((self.width - draw_width) * 0.5, (self.height - draw_height) * 0.5)

    def _to_screen(self, point: Vec2) -> tuple[float, float]:
        world = self.simulation.world
        x = self._arena_offset.x + point.x * self._arena_scale
        # world y grows downward, screen y grows upward
        y = self._arena_offset.y + (world.height - point.y) * self._arena_scale
        return x, y

    def _draw_background(self) -> None:
        world = self.simulation.world
        width = world.width * self._arena_scale
        height = world.height * self._arena_scale
        pyglet.shapes.Rectangle(0, 0, self.width, self.height, color=(6, 7, 9)).draw()
        pyglet.shapes.Rectangle(
            self._arena_offset.x,
            self._arena_offset.y,
            width,
            height,
            color=world.config.background_color,
        ).draw()

    def _draw_entities(self) -> None:
        self._draw_snake()
        self._draw_player()

    def _draw_snake(self) -> None:
        snake = self.simulation.snake
        base_color, pupil_color = self._snake_colors(self.simulation.sense.alert_state)
        total = max(1, len(snake.segments) - 1)
        # draw tail first so the head stays visually on top
        for index in reversed(range(len(snake.segments))):
            segment = snake.segments[index]
            t = index / total
            radius = (snake.body_thickness * (1.0 - 0.58 * t)) * self._arena_scale
            color = (
                int(base_color[0] * (1.0 - 0.3 * t)),
                int(base_color[1] * (1.0 - 0.3 * t)),
                int(base_color[2] * (1.0 - 0.3 * t)),
            )
            sx, sy = self._to_screen(segment)
            circle = pyglet.shapes.Circle(sx, sy, radius, color=color)
            circle.opacity = int(105 + 150 * (1 - t))
            circle.draw()
        self._draw_snake_eye(pupil_color)

    def _snake_colors(self, alert_state: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        if alert_state == "seen":
            return (196, 51, 2), (255, 40, 40)
        if alert_state == "heard":
            return (237, 170, 37), (255, 220, 80)
        return (10, 155, 155), (40, 80, 40)

    def _draw_snake_eye(self, pupil_color: tuple[int, int, int]) -> None:
        snake = self.simulation.snake
        direction = snake.facing_dir
        if direction.length() <= 0.001:
            direction = Vec2(1.0, 0.0)
        head = snake.segments[0] if snake.segments else snake.head
        head_radius = snake.body_thickness * self._arena_scale
        eye_world = Vec2(
            head.x + direction.x * snake.body_thickness * 0.55,
            head.y + direction.y * snake.body_thickness * 0.55,
        )
        eye_x, eye_y = self._to_screen(eye_world)
        pupil_world = Vec2(
            eye_world.x + direction.x * snake.body_thickness * 0.25,
            eye_world.y + direction.y * snake.body_thickness * 0.25,
        )
        pupil_x, pupil_y = self._to_screen(pupil_world)
        pyglet.shapes.Circle(eye_x, eye_y, head_radius * 0.75, color=(245, 247, 250)).draw()
        pyglet.shapes.Circle(pupil_x, pupil_y, head_radius * 0.4, color=pupil_color).draw()

    def _draw_player(self) -> None:
        player = self.simulation.player
        px, py = self._to_screen(player.pos)
        radius = player.radius * self._arena_scale
        pyglet.shapes.Circle(px, py, radius, color=(245, 247, 250)).draw()
        eye_x = px + player.eye_offset.x * self._arena_scale
        eye_y = py - player.eye_offset.y * self._arena_scale
        pyglet.shapes.Circle(eye_x, eye_y, radius * 0.48, color=(12, 14, 18)).draw()
        pyglet.shapes.Circle(eye_x, eye_y, radius * 0.22, color=(92, 210, 255)).draw()


def run() -> None:
    ResearchWindow()
    pyglet.app.run()
