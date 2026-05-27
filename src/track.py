"""Kalman tracker for the ball, with measurement gating.

State:        [x, y, vx, vy]   (position and velocity, pixels and px/frame)
Measurement:  [x, y]           from the detector's bounding-box center

Each frame:
  1. predict() advances the state using the constant-velocity model.
  2. If a measurement is offered, check the distance to the prediction:
       - within MAX_JUMP_PIXELS accept with the measurement
       - too far reject
  3. After too many missed/rejected frames in a row, reset the filter."""

import math
from typing import Optional

import cv2
import numpy as np

# Largest plausible per-frame ball motion in pixels, at this camera zoom.
MAX_JUMP_PIXELS = 150.0


class BallTracker:
    def __init__(
        self,
        reset_after_misses: int = 15,
        max_jump_pixels: float = MAX_JUMP_PIXELS,
    ) -> None:
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32,
        )

        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32,
        )

        # Process noise
        self.kf.processNoiseCov = np.diag([1.0, 1.0, 5.0, 5.0]).astype(np.float32)

        # Measurement noise
        self.kf.measurementNoiseCov = np.diag([4.0, 4.0]).astype(np.float32)

        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1000.0

        self.initialized = False
        self.misses = 0
        self.reset_after = reset_after_misses
        self.max_jump = max_jump_pixels
        # Reporting flags for the pipeline overlay.
        self.last_used_measurement = False
        self.last_rejected_distance: Optional[float] = None

    def reset(self) -> None:
        self.initialized = False
        self.misses = 0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1000.0
        self.last_used_measurement = False
        self.last_rejected_distance = None

    def update(
        self, measurement: Optional[tuple[float, float]]
    ) -> Optional[tuple[float, float]]:
        """Feed an (x, y) measurement or None. Return current best (x, y)
        estimate, or None if not initialized."""

        # First-time initialization needs a real measurement.
        if not self.initialized:
            self.last_used_measurement = False
            self.last_rejected_distance = None
            if measurement is None:
                return None
            x, y = measurement
            self.kf.statePost = np.array(
                [[x], [y], [0], [0]], dtype=np.float32
            )
            self.initialized = True
            self.misses = 0
            self.last_used_measurement = True
            return (x, y)

        # Always advance the motion model.
        predicted = self.kf.predict()
        pred_x, pred_y = float(predicted[0, 0]), float(predicted[1, 0])

        # Gate the measurement.
        accept = False
        self.last_rejected_distance = None
        if measurement is not None:
            mx, my = measurement
            distance = math.hypot(mx - pred_x, my - pred_y)
            if distance <= self.max_jump:
                accept = True
            else:
                self.last_rejected_distance = distance

        if accept:
            self.kf.correct(
                np.array([[np.float32(measurement[0])],
                          [np.float32(measurement[1])]])
            )
            self.misses = 0
            self.last_used_measurement = True
        else:
            self.misses += 1
            self.last_used_measurement = False
            if self.misses > self.reset_after:
                self.reset()
                return None

        return (float(self.kf.statePost[0, 0]),
                float(self.kf.statePost[1, 0]))
