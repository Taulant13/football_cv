"""Convert tracker positions to ball speed in m/s.

SpeedEstimator takes consecutive (x, y) positions from BallTracker and
returns a smoothed speed in meters per second.

Pixel-space velocity is the Euclidean distance between consecutive positions
times the frame rate."""

from collections import deque
import math
from typing import Optional


class SpeedEstimator:
    def __init__(self, px_per_meter: float, fps: int, smooth_window: int = 5) -> None:
        self.ppm = px_per_meter
        self.fps = fps
        self.window: deque = deque(maxlen=smooth_window)
        self.prev: Optional[tuple[float, float]] = None

    def update(self, position: Optional[tuple[float, float]]) -> Optional[float]:
        if position is None:
            self.prev = None
            self.window.clear()
            return None

        if self.prev is None:
            self.prev = position
            return None

        dx = position[0] - self.prev[0]
        dy = position[1] - self.prev[1]
        pixel_dist = math.hypot(dx, dy)
        speed_mps = (pixel_dist / self.ppm) * self.fps

        self.window.append(speed_mps)
        self.prev = position
        return sum(self.window) / len(self.window)
