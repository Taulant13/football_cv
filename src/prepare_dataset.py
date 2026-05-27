"""Build a YOLO-format dataset of ball bounding boxes.

For each sequence in TRAIN_SEQS and VAL_SEQS:
  - parse gameinfo.ini to find the ball's track ID
  - read gt.txt and keep only ball rows
  - subsample every Nth frame
  - copy the image to data/processed/
  - write a YOLO label to data/processed/"""

from pathlib import Path
import configparser
import shutil

import cv2

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "tracking" / "train"
PROC_DIR = ROOT / "data" / "processed"

TRAIN_SEQS = [f"SNMOT-{i:03d}" for i in range(60, 72)]   # 060..071  (12)
VAL_SEQS = ["SNMOT-072", "SNMOT-073"]
TEST_SEQS = ["SNMOT-074", "SNMOT-075"] 

FRAME_STRIDE = 5  # keep every 5th frame


def find_ball_track_id(gameinfo_path: Path) -> int:
    cp = configparser.ConfigParser()
    cp.read(gameinfo_path)
    for key, value in cp["Sequence"].items():
        if key.startswith("trackletid_") and value.strip().lower().startswith("ball"):
            return int(key.split("_")[1])
    raise RuntimeError(f"No ball entry in {gameinfo_path}")


def load_ball_rows(gt_path: Path, ball_id: int):
    rows = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame, tid, x, y, w, h = (int(p) for p in parts[:6])
            if tid == ball_id:
                rows[frame] = (x, y, w, h)
    return rows


def process_sequence(seq_name: str, split: str, img_w: int, img_h: int) -> int:
    seq_dir = RAW_DIR / seq_name
    img_dir = seq_dir / "img1"
    out_img = PROC_DIR / split / "images"
    out_lbl = PROC_DIR / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    ball_id = find_ball_track_id(seq_dir / "gameinfo.ini")
    rows = load_ball_rows(seq_dir / "gt" / "gt.txt", ball_id)

    written = 0
    for frame, (x, y, w, h) in rows.items():
        if frame % FRAME_STRIDE != 0:
            continue
        src_img = img_dir / f"{frame:06d}.jpg"
        if not src_img.exists():
            continue

        dst_name = f"{seq_name}_{frame:06d}"
        shutil.copy2(src_img, out_img / f"{dst_name}.jpg")

        x_c = (x + w / 2) / img_w
        y_c = (y + h / 2) / img_h
        w_n = w / img_w
        h_n = h / img_h
        with open(out_lbl / f"{dst_name}.txt", "w") as f:
            f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
        written += 1

    print(f"  {seq_name} [{split}]: ball_id={ball_id}, wrote {written} frames")
    return written


def read_resolution(seq_dir: Path) -> tuple[int, int]:
    cp = configparser.ConfigParser()
    cp.read(seq_dir / "seqinfo.ini")
    return int(cp["Sequence"]["imWidth"]), int(cp["Sequence"]["imHeight"])


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Train sequences: {TRAIN_SEQS}")
    print(f"Val sequences:   {VAL_SEQS}")
    print(f"Test sequences:  {TEST_SEQS}\n")

    w, h = read_resolution(RAW_DIR / TRAIN_SEQS[0])
    print(f"Frame resolution: {w}x{h}")

    total_train = sum(process_sequence(s, "train", w, h) for s in TRAIN_SEQS)
    total_val = sum(process_sequence(s, "val", w, h) for s in VAL_SEQS)

    print(f"Total train examples: {total_train}")
    print(f"Total val examples:   {total_val}")
    print(f"Output: {PROC_DIR}")


if __name__ == "__main__":
    main()
