# Python Porting Notes

This note records the small, practical port from the old p5.js runtime to the Python research runtime.

## File Mapping

- `archive/frontend/sketch.js` -> `main.py` and `src/pseudoden_research/app.py`
- `archive/frontend/components/Player.js` -> `src/pseudoden_research/entities.py`
- `archive/frontend/components/Hunter.js` -> `src/pseudoden_research/entities.py` and `src/pseudoden_research/behavior.py`
- `archive/frontend/components/Pathfinding.js` -> `src/pseudoden_research/strategies.py`
- `archive/frontend/components/World.js` -> `src/pseudoden_research/world.py`
- `archive/frontend/components/Math.js` -> `src/pseudoden_research/behavior.py` and `src/pseudoden_research/geometry.py`

## Syntax And Function Changes

- p5.js `createVector(x, y)` became the local `Vec2(x, y)` dataclass.
- p5.js vector methods such as `copy()`, `mag()`, `normalize()`, and `dist()` became `Vec2.copy()`, `Vec2.length()`, `Vec2.normalized()`, and `Vec2.distance_to()`.
- p5.js `map(value, a, b, c, d)` became `map_value(value, a, b, c, d)`.
- p5.js `radians()`, `atan2()`, `sin()`, and `cos()` became Python `math.radians`, `math.atan2`, `math.sin`, and `math.cos`.
- p5.js `random()` became a Python `random.Random` instance so the experiment can be repeated with a seed.
- p5.js `deltaTime` in milliseconds became Python `dt` in seconds.
- p5.js drawing calls such as `circle()` and `rect()` became `pyglet.shapes.Circle` and `pyglet.shapes.Rectangle`.
- JavaScript object literals became Python dataclasses for stable experiment data.

## Snake Behavior Port

The old snake was not a direct follower. The port keeps the same behavior order:

1. `SnakeSense.update()` checks hearing and vision.
2. `SnakeMind.update()` chooses a target from the state machine.
3. `AStarStrategy.plan()` finds a path only to that chosen target.
4. `Snake.update()` follows the current path smoothly.

The state machine keeps the original states:

- `IDLE`: slow movement and short delay before patrol.
- `PATROL`: random nearby target.
- `SEARCH`: follows a heard or last-seen position.
- `CHASE`: follows the last seen player position.
- `LOST`: stops briefly before returning to patrol.

The personality parameters from `Math.js` are now represented by `Personality` and mapped into `SnakeConfig`. The important sensory values are:

- `hearing_range`: derived from `Fe`, 10 to 20 grid cells.
- `vision_range`: derived from `Ni`, 5 to 10 grid cells.
- `vision_fov`: derived from `Ti`, 60 to 120 degrees.
- `hearing_confidence_bias`: derived from `Ti` and `Ni`, used for noisy heard positions.

## Removed / Simplified

- Scene switching, tilemaps, sprite sheets, intro UI, gravity, and collision layers are archived with the old frontend runtime.
- The Python runtime uses one solid-color arena and one snake.
- Developer mode UI, path lines, grid overlays, and parameter panels were removed from the visible app.
- Telemetry still records internal state to CSV for research use.
