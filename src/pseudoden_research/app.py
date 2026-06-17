from __future__ import annotations

from math import atan2, cos, degrees, radians, sin

import pyglet
from pyglet.window import key

from .config import TelemetryConfig
from .geometry import Vec2
from .simulation import GameSimulation


MIN_WINDOW_WIDTH = 900
MIN_WINDOW_HEIGHT = 520
# UI
STATUS_PANEL_MIN_WIDTH = 300.0
STATUS_PANEL_MAX_WIDTH = 360.0
STATUS_PANEL_RATIO = 0.25
STATUS_PANEL_PADDING = 20.0
STATUS_TITLE_FONT_SIZE = 17
STATUS_SECTION_FONT_SIZE = 13
STATUS_ROW_FONT_SIZE = 12
STATUS_LINE_GAP = 4.0


class ResearchWindow(pyglet.window.Window):
    def __init__(self) -> None:
        super().__init__(
            width=1440,
            height=810,
            caption="PseudoDen",
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
        self._show_debug_overlay = False
        self._overlay_key_down = False
        self._status_panel_width = 0.0
        pyglet.clock.schedule_interval(self.update, 1.0 / 60.0)

    def update(self, dt: float) -> None:
        # cap large frame gaps so the simulation stays stable
        self.simulation.step(self._read_input(), min(dt, 1.0 / 20.0))

    def on_draw(self) -> None:
        self.clear()
        self._layout_arena()
        self._draw_background()
        self._draw_obstacles()
        self._draw_entities()
        if self._show_debug_overlay:
            self._draw_developer_overlay()
        self._draw_status_panel()

    def on_close(self) -> None:
        self.simulation.close()
        super().on_close()

    # m to open devmode
    def on_key_press(self, symbol: int, modifiers: int) -> object | None:
        if symbol == key.M and not self._overlay_key_down:
            self._show_debug_overlay = not self._show_debug_overlay
            self._overlay_key_down = True
            return pyglet.event.EVENT_HANDLED
        return None

    def on_key_release(self, symbol: int, modifiers: int) -> object | None:
        if symbol == key.M:
            self._overlay_key_down = False
            return pyglet.event.EVENT_HANDLED
        return None

    def _read_input(self) -> Vec2:
        left = self.keys[key.A] or self.keys[key.LEFT]
        right = self.keys[key.D] or self.keys[key.RIGHT]
        up = self.keys[key.W] or self.keys[key.UP]
        down = self.keys[key.S] or self.keys[key.DOWN]
        return Vec2(float(right) - float(left), float(down) - float(up))

    def _layout_arena(self) -> None:
        world = self.simulation.world
        self._status_panel_width = self._get_status_panel_width()
        available_width = max(1.0, self.width - self._status_panel_width)
        scale = min(available_width / world.width, self.height / world.height)
        draw_width = world.width * scale
        draw_height = world.height * scale
        self._arena_scale = scale
        self._arena_offset.set((available_width - draw_width) * 0.5, (self.height - draw_height) * 0.5)

    def _get_status_panel_width(self) -> float:
        return min(STATUS_PANEL_MAX_WIDTH, max(STATUS_PANEL_MIN_WIDTH, self.width * STATUS_PANEL_RATIO))

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
        panel_x = self.width - self._status_panel_width
        pyglet.shapes.Rectangle(
            panel_x,
            0,
            self._status_panel_width,
            self.height,
            color=(10, 12, 16),
        ).draw()
        pyglet.shapes.Line(panel_x, 0, panel_x, self.height, thickness=1.0, color=(44, 50, 62)).draw()

    # draw blocks
    def _draw_obstacles(self) -> None:
        world = self.simulation.world
        cell_size = world.cell_size * self._arena_scale
        for obstacle in world.obstacle_rectangles():
            col_start = max(0, obstacle.col)
            row_start = max(0, obstacle.row)
            col_end = min(world.cols, obstacle.col + obstacle.width)
            row_end = min(world.rows, obstacle.row + obstacle.height)
            if col_start >= col_end or row_start >= row_end:
                continue

            x = self._arena_offset.x + col_start * cell_size
            y = self._arena_offset.y + (world.height - row_end * world.cell_size) * self._arena_scale
            width = (col_end - col_start) * cell_size
            height = (row_end - row_start) * cell_size
            pyglet.shapes.Rectangle(x, y, width, height, color=(34, 42, 52)).draw()
            pyglet.shapes.Line(x, y, x + width, y, thickness=1.0, color=(78, 92, 112)).draw()
            pyglet.shapes.Line(x, y + height, x + width, y + height, thickness=1.0, color=(78, 92, 112)).draw()
            pyglet.shapes.Line(x, y, x, y + height, thickness=1.0, color=(78, 92, 112)).draw()
            pyglet.shapes.Line(x + width, y, x + width, y + height, thickness=1.0, color=(78, 92, 112)).draw()

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

    # draw devmode help lines(hearing zone, destination, etc...)
    def _draw_developer_overlay(self) -> None:
        self._draw_hearing_circle()
        self._draw_vision_arc()
        self._draw_path_route()
        self._draw_target_marker()

    def _draw_hearing_circle(self) -> None:
        snake = self.simulation.snake
        sx, sy = self._to_screen(snake.head)
        radius = self.simulation.snake_config.hearing_range * self._arena_scale
        pyglet.shapes.Circle(sx, sy, radius, segments=96, color=(92, 210, 255, 18)).draw()
        pyglet.shapes.Arc(sx, sy, radius, segments=96, thickness=2.0, color=(92, 210, 255, 145)).draw()

    def _draw_vision_arc(self) -> None:
        snake = self.simulation.snake
        direction = snake.facing_dir.normalized()
        if direction.length() <= 0.001:
            direction = Vec2(1.0, 0.0)

        sx, sy = self._to_screen(snake.head)
        radius = self.simulation.snake_config.vision_range * self._arena_scale
        fov_degrees = degrees(self.simulation.snake_config.vision_fov)
        center_degrees = degrees(atan2(-direction.y, direction.x))
        start_degrees = center_degrees - fov_degrees * 0.5
        end_degrees = center_degrees + fov_degrees * 0.5

        pyglet.shapes.Arc(
            sx,
            sy,
            radius,
            segments=64,
            angle=fov_degrees,
            start_angle=start_degrees,
            thickness=3.0,
            color=(255, 196, 87, 210),
        ).draw()
        for angle_degrees in (start_degrees, end_degrees):
            angle = radians(angle_degrees)
            ex = sx + cos(angle) * radius
            ey = sy + sin(angle) * radius
            pyglet.shapes.Line(sx, sy, ex, ey, thickness=1.5, color=(255, 196, 87, 135)).draw()

    def _draw_path_route(self) -> None:
        metrics = self.simulation.last_metrics
        points = metrics.decision.points if metrics else self.simulation.snake.path_points
        if not points:
            return

        route = [self.simulation.snake.head, *points]
        for left, right in zip(route, route[1:]):
            x1, y1 = self._to_screen(left)
            x2, y2 = self._to_screen(right)
            pyglet.shapes.Line(x1, y1, x2, y2, thickness=3.0, color=(81, 220, 132, 215)).draw()

        marker_radius = max(2.5, 4.5 * self._arena_scale)
        for point in points[1:]:
            px, py = self._to_screen(point)
            pyglet.shapes.Circle(px, py, marker_radius, color=(120, 255, 170, 205)).draw()

    def _draw_target_marker(self) -> None:
        target = self.simulation.mind.current_target
        if not target and self.simulation.last_metrics:
            target = self.simulation.last_metrics.decision.target
        if not target:
            return

        tx, ty = self._to_screen(target)
        radius = 9.0
        pyglet.shapes.Circle(tx, ty, radius, color=(255, 82, 114, 230)).draw()
        pyglet.shapes.Line(tx - radius * 1.6, ty, tx + radius * 1.6, ty, thickness=2.0, color=(255, 235, 238)).draw()
        pyglet.shapes.Line(tx, ty - radius * 1.6, tx, ty + radius * 1.6, thickness=2.0, color=(255, 235, 238)).draw()

    # right UI(jung functions and state machine)
    def _draw_status_panel(self) -> None:
        x = self.width - self._status_panel_width + STATUS_PANEL_PADDING
        y = self.height - STATUS_PANEL_PADDING
        content_width = int(self._status_panel_width - STATUS_PANEL_PADDING * 2)
        personality = self.simulation.personality
        mind = self.simulation.mind
        sense = self.simulation.sense

        y = self._draw_panel_line(
            "Snake Status",
            x,
            y,
            font_size=STATUS_TITLE_FONT_SIZE,
            color=(244, 247, 252, 255),
            width=content_width,
        )
        y -= 12
        y = self._draw_panel_section(
            "Jung Functions",
            [
                f"Se {personality.Se}    Si {personality.Si}",
                f"Ne {personality.Ne}    Ni {personality.Ni}",
                f"Te {personality.Te}    Ti {personality.Ti}",
                f"Fe {personality.Fe}    Fi {personality.Fi}",
            ],
            x,
            y,
            content_width,
        )
        self._draw_panel_section(
            "State Machine",
            [
                f"State: {mind.state}",
                f"Alert: {sense.alert_state}",
                f"State time: {mind.state_time:.2f}s",
                f"Chase time: {mind.chase_time:.2f}s",
                f"Target: {self._format_vec(mind.current_target)}",
            ],
            x,
            y,
            content_width,
        )

    def _draw_panel_section(self, title: str, lines: list[str], x: float, y: float, width: int) -> float:
        y -= 4
        pyglet.shapes.Line(x, y, x + width, y, thickness=1.0, color=(44, 50, 62)).draw()
        y -= 10
        y = self._draw_panel_line(
            title,
            x,
            y,
            font_size=STATUS_SECTION_FONT_SIZE,
            color=(117, 205, 255, 255),
            width=width,
        )
        for line in lines:
            y = self._draw_panel_line(line, x, y, color=(199, 208, 223, 255), width=width)
        return y - 8

    def _draw_panel_line(
        self,
        text: str,
        x: float,
        y: float,
        font_size: int = STATUS_ROW_FONT_SIZE,
        color: tuple[int, int, int, int] = (226, 232, 240, 255),
        width: int | None = None,
    ) -> float:
        label = pyglet.text.Label(
            text,
            x=x,
            y=y,
            width=width,
            multiline=False,
            anchor_x="left",
            anchor_y="top",
            font_name="Consolas",
            font_size=font_size,
            color=color,
        )
        label.draw()
        return y - (font_size + STATUS_LINE_GAP)

    def _format_vec(self, point: Vec2 | None) -> str:
        if not point:
            return "None"
        return f"{point.x:.1f}, {point.y:.1f}"


def run() -> None:
    ResearchWindow()
    pyglet.app.run()
