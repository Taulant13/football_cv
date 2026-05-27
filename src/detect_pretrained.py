"""Naive ball detection with pretrained YOLOv8 (no fine-tuning).

Runs the YOLOv8 model on every frame of one sequence, filters
to COCO class 32 ('sports ball'), draws the detections, and writes the result
as outputs/01_pretrained.mp4."""

from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = ROOT / "data" / "raw" / "tracking" / "train" / "SNMOT-060"
IMG_DIR = SEQ_DIR / "img1"
OUT_PATH = ROOT / "outputs" / "01_pretrained.mp4"

MODEL_NAME = "yolov8s.pt"     
COCO_SPORTS_BALL_CLASS = 32
CONF_THRESHOLD = 0.10            # lower than default to give pretrained model a chance
FPS = 25


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(IMG_DIR.glob("*.jpg"))
    if not frame_paths:
        raise SystemExit(f"No frames found in {IMG_DIR}")

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (w, h))

    model = YOLO(MODEL_NAME)

    detections_count = 0
    frames_with_detection = 0

    for fp in tqdm(frame_paths, desc="pretrained detect"):
        img = cv2.imread(str(fp))
        results = model.predict(
            img,
            classes=[COCO_SPORTS_BALL_CLASS],
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        boxes = results[0].boxes
        n = 0 if boxes is None else len(boxes)
        if n > 0:
            frames_with_detection += 1
            detections_count += n
            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    img, f"ball {conf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                )

        cv2.putText(
            img, "pretrained YOLOv8 (baseline)", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
        )
        writer.write(img)

    writer.release()

    print(f"Frames processed:          {len(frame_paths)}")
    print(f"Frames with a detection:   {frames_with_detection}")
    print(f"Total detections:          {detections_count}")
    print(f"Detection coverage:        {frames_with_detection / len(frame_paths):.1%}")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
