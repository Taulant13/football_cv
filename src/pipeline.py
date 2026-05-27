"""End-to-end pipeline: detect + pick + track + draw.
  1. DETECT  - fine-tuned YOLOv8 produces 0-N candidate boxes.
  2. PICK    - keep the highest-confidence box (or None if no detection).
  3. TRACK   - Kalman filter smooths the position and coasts through misses.
  4. DRAW    - fading trajectory line, ball marker, status overlay.

Stages 1-2 are stateless. Stages 3-4 carry state across frames."""

from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from track import BallTracker

ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = ROOT / "data" / "raw" / "tracking" / "train" / "SNMOT-060"
IMG_DIR = SEQ_DIR / "img1"
OUT_PATH = ROOT / "outputs" / "03_trajectory.mp4"
WEIGHTS = ROOT / "models" / "yolov8_ball.pt"

CONF_THRESHOLD = 0.20
FPS = 25
TRAIL_LENGTH = 30           # how many past positions to draw
RESET_AFTER_MISSES = 15     # reset Kalman after this many missed frames


def pick_best_detection(boxes):
    """Return the (x, y) center of the highest-confidence ball box, or None."""
    if boxes is None or len(boxes) == 0:
        return None
    confs = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def draw_trail(img, trail: deque) -> None:
    """Draw a fading polyline from oldest to newest position."""
    pts = list(trail)
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        thickness = max(1, int(1 + 4 * alpha))
        color = (0, int(255 * alpha), int(255 * (1 - alpha)))  # blue -> green
        p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p2 = (int(pts[i][0]), int(pts[i][1]))
        cv2.line(img, p1, p2, color, thickness)


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"Weights not found: {WEIGHTS}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(IMG_DIR.glob("*.jpg"))
    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(OUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h)
    )

    model = YOLO(str(WEIGHTS))
    tracker = BallTracker(reset_after_misses=RESET_AFTER_MISSES)
    trail: deque = deque(maxlen=TRAIL_LENGTH)

    raw_hits = 0
    accepted = 0
    rejected = 0
    tracked_frames = 0

    for fp in tqdm(frame_paths, desc="pipeline"):
        img = cv2.imread(str(fp))
        results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)
        measurement = pick_best_detection(results[0].boxes)

        if measurement is not None:
            raw_hits += 1

        position = tracker.update(measurement)
        if tracker.last_used_measurement:
            accepted += 1
        elif tracker.last_rejected_distance is not None:
            rejected += 1

        if position is not None:
            tracked_frames += 1
            trail.append(position)
            draw_trail(img, trail)
            cx, cy = int(position[0]), int(position[1])
            cv2.circle(img, (cx, cy), 8, (0, 255, 0), 2)

        if tracker.last_used_measurement:
            status, color = "DETECTED", (0, 255, 0)
        elif tracker.last_rejected_distance is not None:
            status, color = (
                f"rejected (d={tracker.last_rejected_distance:.0f}px)",
                (0, 0, 255),
            )
        else:
            status, color = "predicted", (0, 165, 255)
        cv2.putText(
            img, f"fine-tuned + Kalman  [{status}]", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
        )

        writer.write(img)

    writer.release()

    n = len(frame_paths)
    print(f"Frames processed:                {n}")
    print(f"Frames with raw detection:       {raw_hits}  ({raw_hits / n:.1%})")
    print(f"Detections accepted by tracker:  {accepted}")
    print(f"Detections rejected by gating:   {rejected}")
    print(f"Frames with tracked position:    {tracked_frames}  ({tracked_frames / n:.1%})")
    print(f"Frames filled by Kalman alone:   {tracked_frames - accepted}")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
