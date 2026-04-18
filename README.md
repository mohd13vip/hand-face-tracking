CV Competition Submission - ParrotCyber

This zip contains:
- Task1_ChallengeB: YOLO model training + results + report
- Task2_Tracking: Tracking outputs + metrics + videos + report

How to run:
1) Install dependencies: pip install -r requirements.txt
2) Task1: see Task1_ChallengeB folder
3) Task2: see Task2_Tracking folder

# Computer Vision Competition Submission (Task 1 + Task 2)

Team submission containing:
- **Task 1 (Challenge B): Efficient Vision System Design** — train and compare YOLOv8n / YOLOv8s / YOLOv8m and justify the best trade‑off.
- **Task 2 (Tracking Challenge): Multi‑Object Tracking on MOT17** — YOLO detection + ByteTrack tracking, export MOT-format tracks, compute metrics, and generate demo videos.

## Folder Structure (what you will upload)
This repository/zip is expected to have this structure:

```
final_submission/
├── README.md
├── requirements.txt
├── setup_guide.md
├── verify_setup.py
├── task1_challengeB/
│   ├── code/
│   ├── models/
│   ├── results/
│   ├── examples/
│   └── Challenge_B_Report.(pdf|docx)
└── task2_tracking/
    ├── code/
    ├── results/
    │   ├── tracks/                 # MOT-format .txt outputs (one per sequence)
    │   ├── metrics/                # tracking_metrics.csv
    │   └── videos/                 # .mp4 visualization outputs
    └── Tracking_Report.(pdf|docx)
```

## Quick Start
### 1) Create environment
```bash
python -m venv yolo_env
# Windows
yolo_env\Scripts\activate
pip install -r requirements.txt
```

### 2) Verify setup
```bash
python verify_setup.py
```

## Task 1: Challenge B (YOLO comparison)
From the project root:
```bash
cd task1_challengeB
# Train (if needed) + evaluate/benchmark
python code\benchmark.py
```
Outputs should include:
- model weights in `task1_challengeB/models/`
- comparison CSV + charts in `task1_challengeB/results/`
- example detections in `task1_challengeB/examples/`

## Task 2: Tracking Challenge (YOLO + ByteTrack)
From the project root:
```bash
cd task2_tracking
# Run tracking / evaluation (adjust according to your scripts)
python code\main_tracker.py
python code\evaluate.py
```

To generate demo videos (example):
```bash
python code\make_video_from_tracks.py --seq datasets\MOT17\train\MOT17-02-DPM --tracks results\tracks\MOT17-02-DPM.txt --out results\videos\MOT17-02-DPM.mp4
```

## Notes
- Reports can be `.docx` or exported to `.pdf` (preferred if the form accepts PDFs).
- If your zip becomes very large due to videos, keep only the required demo sequences (or compress videos).
