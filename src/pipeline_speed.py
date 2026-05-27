"""Pipeline with live speed overlay.

Same detect -> pick -> track -> draw chain as pipeline.py, plus a new stage:
  - SpeedEstimator converts tracker positions to a smoothed m/s readout
  - The current speed is displayed in the top-left of every frame

Reads the pixel-to-meter scale from data/scale.json."""

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from track import BallTracker
from speed import SpeedEstimator

ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = ROOT / "data" / "raw" / "tracking" / "train" / "SNMOT-060"
IMG_DIR = SEQ_DIR / "img1"
OUT_PATH = ROOT / "outputs" / "04_speed.mp4"
WEIGHTS = ROOT / "models" / "yolov8_ball.pt"
SCALE_PATH = ROOT / "data" / "scale.json"

CONF_THRESHOLD = 0.20
FPS = 25
TRAIL_LENGTH = 30
RESET_AFTER_MISSES = 15
SPEED_SMOOTH_WINDOW = 5


def pick_best_detection(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    confs = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def draw_trail(img, trail: deque) -> None:
    pts = list(trail)
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        thickness = max(1, int(1 + 4 * alpha))
        color = (0, int(255 * alpha), int(255 * (1 - alpha)))
        p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p2 = (int(pts[i][0]), int(pts[i][1]))
        cv2.line(img, p1, p2, color, thickness)


def load_scale() -> float:
    if not SCALE_PATH.exists():
        raise SystemExit(
            f"Calibration file not found\n"
            "Run `python src/calibrate_scale.py` first to set the pixel-to-meter scale."
        )
    cfg = json.loads(SCALE_PATH.read_text())
    ppm = float(cfg["px_per_meter"])
    print(f"Loaded scale: {ppm:.3f} pixels per meter (from {SCALE_PATH.name})")
    return ppm


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"Weights not found: {WEIGHTS}")

    px_per_meter = load_scale()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(IMG_DIR.glob("*.jpg"))
    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(OUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h)
    )

    model = YOLO(str(WEIGHTS))
    tracker = BallTracker(reset_after_misses=RESET_AFTER_MISSES)
    speed_est = SpeedEstimator(px_per_meter=px_per_meter, fps=FPS,
                                smooth_window=SPEED_SMOOTH_WINDOW)
    trail: deque = deque(maxlen=TRAIL_LENGTH)

    peak_speed = 0.0
    speeds: list[float] = []

    for fp in tqdm(frame_paths, desc="speed pipeline"):
        img = cv2.imread(str(fp))
        results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)
        measurement = pick_best_detection(results[0].boxes)

        position = tracker.update(measurement)
        speed_mps = speed_est.update(position)

        if position is not None:
            trail.append(position)
            draw_trail(img, trail)
            cx, cy = int(position[0]), int(position[1])
            cv2.circle(img, (cx, cy), 8, (0, 255, 0), 2)

        # status banner (top-left)
        if tracker.last_used_measurement:
            status, color = "DETECTED", (0, 255, 0)
        elif tracker.last_rejected_distance is not None:
            status, color = (
                f"rejected (d={tracker.last_rejected_distance:.0f}px)",
                (0, 0, 255),
            )
        else:
            status, color = "predicted", (0, 165, 255)
        cv2.putText(img, f"fine-tuned + Kalman  [{status}]", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # speed readout (below status)
        if speed_mps is not None:
            speeds.append(speed_mps)
            peak_speed = max(peak_speed, speed_mps)
            speed_kmh = speed_mps * 3.6
            cv2.putText(img, f"speed: {speed_mps:5.1f} m/s  ({speed_kmh:5.1f} km/h)",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        writer.write(img)

    writer.release()

    n = len(frame_paths)
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    print(f"Frames processed:       {n}")
    print(f"Frames with speed:      {len(speeds)}")
    print(f"Average ball speed:     {avg_speed:.2f} m/s  ({avg_speed * 3.6:.2f} km/h)")
    print(f"Peak ball speed:        {peak_speed:.2f} m/s  ({peak_speed * 3.6:.2f} km/h)")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
