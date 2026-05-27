"""Interactive pixel-to-meter calibration.

Opens the first frame of SNMOT-060. Click two points whose real-world
distance is known, then type the distance in meters at the prompt. The script
saves data/scale.json with the resulting pixels-per-meter value."""

import json
import math
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
FRAME_PATH = (
    ROOT / "data" / "raw" / "tracking" / "train" / "SNMOT-060" / "img1" / "000001.jpg"
)
OUT_PATH = ROOT / "data" / "scale.json"


def main() -> None:
    img = cv2.imread(str(FRAME_PATH))
    if img is None:
        raise SystemExit(f"Could not load frame: {FRAME_PATH}")
    display = img.copy()

    clicks: list[tuple[int, int]] = []

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(clicks) >= 2:
            return
        clicks.append((x, y))
        cv2.circle(display, (x, y), 6, (0, 255, 0), 2)
        cv2.putText(display, str(len(clicks)), (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if len(clicks) == 2:
            cv2.line(display, clicks[0], clicks[1], (0, 255, 0), 2)
        cv2.imshow("calibrate", display)

    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("calibrate", 1280, 720)
    cv2.setMouseCallback("calibrate", on_click)
    cv2.imshow("calibrate", display)

    print("Click two points on the image whose real-world distance you know.")
    print("Then close the window or press any key.\n")

    while True:
        if len(clicks) >= 2:
            cv2.waitKey(500)
            break
        if cv2.waitKey(50) & 0xFF == 27:  # ESC
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled.")
    cv2.destroyAllWindows()

    (x1, y1), (x2, y2) = clicks
    pixel_dist = math.hypot(x2 - x1, y2 - y1)
    print(f"Pixel distance: {pixel_dist:.2f}")

    while True:
        try:
            meters = float(input("Real-world distance in meters: ").strip())
            if meters <= 0:
                raise ValueError
            break
        except ValueError:
            print("  Please type a positive number.")

    px_per_meter = pixel_dist / meters
    print(f"\nPixels per meter: {px_per_meter:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "frame": str(FRAME_PATH.name),
        "points": clicks,
        "real_distance_m": meters,
        "pixel_distance": pixel_dist,
        "px_per_meter": px_per_meter,
    }, indent=2))
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
