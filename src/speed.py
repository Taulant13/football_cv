"""Convert tracker positions to ball speed in m/s.

SpeedEstimator takes consecutive (x, y) positions from BallTracker and a
per-frame pixels-per-meter scale, and returns a smoothed speed in m/s.

The scale comes from BallSizeScale, which derives pixels-per-meter from the
detected ball's box width (a regulation ball is ~0.22 m across). Measuring it
per frame adapts automatically to camera zoom and to perspective, so no manual
calibration is needed. The raw estimate is noisy on a ~10 px ball and inflated
by motion blur, so it is taken as a median over a short window and carried
forward when no usable detection is available."""

from collections import deque
import math
import statistics
from typing import Optional


class SpeedEstimator:
    def __init__(self, fps: int, smooth_window: int = 5) -> None:
        self.fps = fps
        self.window: deque = deque(maxlen=smooth_window)
        self.prev: Optional[tuple[float, float]] = None

    def update(
        self,
        position: Optional[tuple[float, float]],
        ppm: Optional[float],
    ) -> Optional[float]:
        if position is None:
            self.prev = None
            self.window.clear()
            return None

        if self.prev is None:
            self.prev = position
            return None

        if ppm is None or ppm <= 0:
            # No usable scale yet (before the first detection). Advance the
            # position but emit no number.
            self.prev = position
            return None

        dx = position[0] - self.prev[0]
        dy = position[1] - self.prev[1]
        pixel_dist = math.hypot(dx, dy)
        speed_mps = (pixel_dist / ppm) * self.fps

        self.window.append(speed_mps)
        self.prev = position
        return sum(self.window) / len(self.window)


class BallSizeScale:
    """Estimate pixels-per-meter from the detected ball's box width.

    A regulation ball is ~0.22 m across, so box_width_px / 0.22 is a local
    pixels-per-meter estimate at the ball's position. A median over a short
    window tames the per-frame noise; the last value is carried forward when
    no usable detection is offered (scale changes slowly vs. occlusions)."""

    def __init__(self, ball_diameter_m: float = 0.22, window: int = 15) -> None:
        self.diameter = ball_diameter_m
        self.samples: deque = deque(maxlen=window)
        self.last: Optional[float] = None

    def update(self, box_width_px: Optional[float]) -> Optional[float]:
        if box_width_px is not None and box_width_px > 0:
            self.samples.append(box_width_px / self.diameter)
        if self.samples:
            self.last = statistics.median(self.samples)
        return self.last
