# Soccer Ball Detection and Trajectory Analysis

Course project (HSLU CVAI). A four-stage pipeline that takes a SoccerNet broadcast clip and produces an annotated video showing the ball detected per frame, its trajectory over time, and its instantaneous speed in m/s.

## Results

| System | Detection coverage | Notes |
|---|---|---|
| Pretrained YOLOv8s (COCO `sports ball`) | 39.9% | Baseline, ball too small / under-represented in COCO |
| **Fine-tuned YOLOv8s** | **95.5%** | mAP50 = 0.663 |
| Fine-tuned + Kalman tracker | 99.7% | 43/750 frames filled by motion prediction |

Measured on `SNMOT-060` (750 frames, 25 fps, 1920×1080).

## Pipeline

```
   raw frame
      │
      ▼
   [1] DETECT       fine-tuned YOLOv8s finds the ball
      │
      ▼
   [2] PICK         keep highest-confidence detection
      │
      ▼
   [3] TRACK        Kalman filter (constant-velocity) + measurement gating
      │
      ▼
   [4] DRAW         trajectory trail, ball marker, speed overlay
      │
      ▼
   annotated frame
```

Stages 1-2 are stateless per-frame. Stages 3-4 carry state across frames.

### The tracker

The Kalman filter holds 4 numbers: `[x, y, vx, vy]`. Each frame it predicts the next position, then corrects with the new measurement. Velocity is never measured directly but inferred over time.

**Measurement gating.** Naïve Kalman blindly trusts whatever the detector hands it. In practice the fine-tuned detector occasionally fires a high-confidence false positive on a corner flag or shin guard, several hundred pixels from the real ball. Without gating, the tracker accepts the teleport and downstream speed estimates explode. The fix is a single threshold: reject any measurement whose distance from the filter's prediction exceeds `MAX_JUMP_PIXELS = 150`. A legitimate 30 m/s shot at that camera zoom moves the ball ~50 px/frame, so the threshold is generous for real motion and tight against teleports.

After 15 consecutive missed or rejected frames the tracker resets. Otherwise stale predictions drift.

### Speed estimation

Consecutive tracker positions give pixel velocity. A one-time 2-click calibration (`calibrate_scale.py`) measures pixels-per-meter against a known feature in the frame (e.g. penalty spot - goalline = 11 m). Multiply by frame rate, smooth with a 5-frame moving average.

## Repository layout

```
football_cv/
├── README.md
├── requirements.txt
├── data/
│   ├── ball.yaml       YOLOv8 dataset config
│   └── scale.json      Calibrated px-per-meter (SNMOT-060)
├── models/
│   └── yolov8_ball.pt  Fine-tuned weights (22 MB)
└── src/
    ├── prepare_dataset.py      MOT labels → YOLO format
    ├── detect_pretrained.py    COCO baseline
    ├── detect_finetuned.py     fine-tuned detector run
    ├── track.py                Kalman tracker (gating + reset)
    ├── pipeline.py             detect -> pick -> track -> draw
    ├── pipeline_speed.py       pipeline.py + speed overlay
    ├── speed.py                Pixel → m/s converter
    └── calibrate_scale.py      Interactive 2-click calibration
```

## Setup

Windows + Python 3.11. PyTorch must be installed separately against the CUDA index, then everything else:

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```


## Running the pipeline

All scripts are run as top-level modules from the project root:

```
python src/detect_pretrained.py      baseline creates outputs/01_pretrained.mp4
python src/prepare_dataset.py        MOT, YOLO under data/processed/
yolo detect train data=data/ball.yaml model=yolov8s.pt imgsz=1280 epochs=30 batch=8
python src/detect_finetuned.py       outputs/02_finetuned.mp4
python src/pipeline.py               outputs/03_trajectory.mp4
python src/calibrate_scale.py        prereq interactive 2-click writes data/scale.json
python src/pipeline_speed.py         outputs/04_speed.mp4
```

A pre-trained `yolov8_ball.pt` is already shipped with this repo, so you can just run inference.

## Data

The training data is from [SoccerNet-Tracking](https://github.com/SoccerNet/sn-tracking), specifically sequences `SNMOT-060` through `SNMOT-075` of the train split (1920×1080 @ 25 fps, 750 frames each).

**The raw data is not in this repository.** SoccerNet's license forbids redistribution, access requires registering, accepting the NDA, and obtaining a password from the SoccerNet team. To reproduce:

1. Register at https://www.soccer-net.org/ and request the tracking password.
2. Download the tracking train split with the SoccerNet DevKit:
   ```python
   from SoccerNet.Downloader import SoccerNetDownloader
   d = SoccerNetDownloader(LocalDirectory="data/raw")
   d.password = "<your password>"
   d.downloadDataTask(task="tracking", split=["train"])
   ```
3. Unzip the downloaded bundles so each sequence sits at `data/raw/tracking/train/SNMOT-XXX/` (with the usual `img1/`, `gt/gt.txt`, `gameinfo.ini`, `seqinfo.ini` inside).

**Split convention** Train SNMOT-060..071, val SNMOT-072..073, test (never touched) SNMOT-074..075. Split is by sequence, not by frame, to prevent near-duplicate leakage. `prepare_dataset.py` also subsamples every 5th frame for the same reason.

**Per-sequence ball ID.** The ball's `trackletID` varies by sequence so read it from `gameinfo.ini` (key `trackletID_<n> = ball`).

## Design invariants

A few choices:

- **`imgsz=1280` for training and inference.**
- **Single-target tracker by design.** The picker reduces N detections to one by argmax(confidence).
- **Measurement gating defends the tracker, not the detector.**
- **Reset after 15 missed frames.** Larger means stale predictions drift, smaller lets brief occlusions break the trail.
- **Inference confidence threshold = 0.20.** Low because the ball is small, the gated tracker tolerates the resulting false positives.

## Limitations

- Single broadcast camera only
- Speed scale is calibrated once per camera zoom from a 2-click feature, report numbers as **relative**, not broadcast-grade absolute.
