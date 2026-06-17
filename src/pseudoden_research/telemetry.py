from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

FIELDNAMES = [
    "timestamp",
    "frame",
    "algorithm",
    "player_x",
    "player_y",
    "snake_x",
    "snake_y",
    "distance",
    "snake_speed",
    "target_x",
    "target_y",
    "path_nodes",
    "path_distance",
    "caught",
    "snake_state",
    "alert_state",
    "hearing_range",
    "vision_range",
]

@dataclass
class TelemetryWriter:
    directory: Path
    interval: float
    enabled: bool = True
    file_path: Path | None = None
    _file: object | None = field(default=None, init=False, repr=False)
    _writer: csv.DictWriter | None = field(default=None, init=False, repr=False)
    _time_since_write: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # timestamped file keeps each run separate
        self.file_path = self.directory / f"astar_baseline_{stamp}.csv"
        self._file = self.file_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()

    def write(self, row: dict[str, object], dt: float, force: bool = False) -> None:
        if not self.enabled or not self._writer:
            return
        self._time_since_write += dt
        # write at a fixed cadence unless an event needs flushing now
        if not force and self._time_since_write < self.interval:
            return
        self._time_since_write = 0.0
        self._writer.writerow(row)
        if self._file:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
