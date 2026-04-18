# Real-time Person, Hand & Face Tracking

Real-time multi-target detection and tracking system built with **YOLOv8** and **MediaPipe**. Detects every person in the frame, assigns each a unique ID and color-coded bounding box, and overlays full hand, finger, and face landmarks.

Originally built for the **ASU Computer Vision Competition v1** (Jan 2026) as Team **ParrotCyber**, then extended with hand, finger, and face tracking on top of the core person-tracking task.

---

## ✨ Features

- **Person detection + tracking** — colored bounding box with a persistent unique ID for every person in frame
- **Hand & finger landmarks** — 21 keypoints per hand (via MediaPipe)
- **Face landmarks** — facial keypoint overlay
- **iVCam support** — use your phone as a 1080p @ 60fps webcam
- **Multi-camera** — laptop webcam, iVCam, or DroidCam
- **Cross-platform** — Windows `.bat` and macOS `.sh` launchers included
- **Competition reference code** — full reports and results for PASCAL VOC detection and MOT17 tracking

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/mohd13vip/hand-face-tracking.git
cd hand-face-tracking
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

**Windows — laptop webcam:**
```bash
run_laptop_cam.bat
```

**Windows — iVCam (phone as webcam):**
```bash
run_ivcam.bat
```

**Run the main script directly:**
```bash
python person_hand_finger_both_ivcam_enhanced.py
```

**macOS:**
```bash
bash run_builtin_cam_mac.sh
```

> The YOLOv8 and MediaPipe model files are downloaded automatically on first run.

---

## 📂 Project Structure

```
.
├── person_hand_finger_both_ivcam_enhanced.py   ⭐ Main script
├── person_hand_finger_both_ivcam.py            iVCam baseline
├── person_hand_finger_both.py                  Generic webcam version
├── person_hand_finger_mac.py                   macOS version
├── webcam_demo.py                              Minimal demo
├── cam_test.py / find_camera.py / ...          Camera discovery utilities
├── requirements.txt
├── START.bat / START_mac.sh                    One-click launchers
│
├── Task1_ChallengeB/                           ASU Competition — Object Detection
│   ├── Challenge_B_Report.pdf                  Full report
│   └── code/, results/, examples/              Code, metrics, sample predictions
│
└── Task2_Tracking/                             ASU Competition — Multi-Object Tracking
    ├── Tracking_Report.pdf                     Full report
    └── code/, results/                         Code, tracking metrics, per-sequence tracks
```

---

## 🛠 Tech Stack

- **Python 3.11**
- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** — person detection and pose estimation
- **[MediaPipe](https://developers.google.com/mediapipe)** — hand, face, and body landmarks
- **OpenCV** — video capture and rendering
- **iVCam / DroidCam** — smartphone-as-webcam support

---

## 🏆 ASU Computer Vision Competition v1 (Jan 2026)

Built with **Team ParrotCyber**:

- **Task 1 — Challenge B: Object Detection**
  PASCAL VOC object detection comparing YOLOv8n / v8s / v8m. Full methodology, training setup, and benchmark results in [`Task1_ChallengeB/Challenge_B_Report.pdf`](Task1_ChallengeB/Challenge_B_Report.pdf).

- **Task 2 — Multi-Object Tracking**
  MOT17 pedestrian tracking. Full results and per-sequence metrics in [`Task2_Tracking/Tracking_Report.pdf`](Task2_Tracking/Tracking_Report.pdf).

---

## 👤 Author

**Mohammed Fadi Abdallah**
AI & Data Science Student — Applied Science University, Jordan
CTF Player — Team Captain, Ded_Sec

- 🔗 LinkedIn: [mohammedfadi-abdallah](https://www.linkedin.com/in/mohammedfadi-abdallah)
- 🐙 GitHub: [@mohd13vip](https://github.com/mohd13vip)
- 📧 mohd13vip19102007@gmail.com

---

## 📜 License

Educational and portfolio use. Contact the author for other uses.
