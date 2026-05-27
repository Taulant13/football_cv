"""Run fine-tuned YOLOv8 on SNMOT-060

This is the after to detect_pretrained.py's before. Same code path, only
the weights and the labels in the overlay differ. The output is written to
outputs/02_finetuned.mp4 and the printout reports detection coverage so we
can directly compare."""

from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = ROOT / "data" / "raw" / "tracking" / "train" / "SNMOT-060"
IMG_DIR = SEQ_DIR / "img1"
OUT_PATH = ROOT / "outputs" / "02_finetuned.mp4"
WEIGHTS = ROOT / "models" / "yolov8_ball.pt"

CONF_THRESHOLD = 0.10
FPS = 25


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(
            f"Weights not found: {WEIGHTS}\n"
            "Copy your fine-tuned best.pt to that path first."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(IMG_DIR.glob("*.jpg"))
    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(OUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h)
    )

    model = YOLO(str(WEIGHTS))

    detections_count = 0
    frames_with_detection = 0

    for fp in tqdm(frame_paths, desc="fine-tuned detect"):
        img = cv2.imread(str(fp))
        results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        n = 0 if boxes is None else len(boxes)
        if n > 0:
            frames_with_detection += 1
            detections_count += n
            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img, f"ball {conf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

        cv2.putText(
            img, "fine-tuned YOLOv8", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
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
