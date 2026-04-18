# Enhanced CV Tracking — macOS Setup Guide

## Requirements

- macOS 12 Monterey or later (Apple Silicon M1/M2/M3 recommended)
- Python 3.11 installed (`brew install python@3.11` if needed)
- iVCam app (optional) for using an iPhone as a webcam

---

## Quick Start

### Step 1 — Run setup (once)

```bash
chmod +x setup_mac.sh START_mac.sh run_ivcam_mac.sh run_builtin_cam_mac.sh
./setup_mac.sh
```

This will:
- Create a `yolo_env311` virtual environment with Python 3.11
- Install PyTorch with MPS support (no CUDA index URL needed)
- Install mediapipe, ultralytics, opencv-python, numpy, scipy, etc.
- Download `yolov8s-pose.pt`
- Verify the installation

### Step 2 — Launch

```bash
./START_mac.sh
```

Select mode:
- **1** — iVCam iPhone camera at 1920×1080
- **2** — Built-in Mac camera at 1280×720

---

## Individual Launch Scripts

| Script | Camera | Resolution |
|--------|--------|------------|
| `run_ivcam_mac.sh` | iVCam (index 0) | 1920×1080 |
| `run_builtin_cam_mac.sh` | Built-in (index 0) | 1280×720 |
| `run_builtin_cam_mac.sh 1` | Built-in (index 1) | 1280×720 |

---

## Camera Index Probing

If you're unsure which index your camera uses, run this one-liner:

```bash
python3 -c "
import cv2
for i in range(6):
    c = cv2.VideoCapture(i)
    if c.isOpened():
        ok, f = c.read()
        status = f'WORKS ({f.shape[1]}x{f.shape[0]})' if (ok and f is not None) else 'no frame'
        print(f'  index {i}: {status}')
    c.release()
"
```

Common layout on MacBook with iVCam connected:
- `index 0` — iVCam (if opened first by the OS)
- `index 1` — Built-in FaceTime camera

---

## MPS vs CUDA

This project uses **Apple MPS** (Metal Performance Shaders) instead of CUDA:

| Windows (original) | macOS (this version) |
|--------------------|----------------------|
| `CUDA_VISIBLE_DEVICES` env var | Removed |
| `torch.cuda.is_available()` | `torch.backends.mps.is_available()` |
| `device = "cuda"` | `device = "mps"` |
| `torch.cuda.*` calls | `torch.backends.mps.*` equivalents |
| YOLO `device="cuda"` | YOLO `device="mps"` |

MediaPipe (hands, face) runs on CPU automatically — no changes needed.

If MPS is not available (Intel Mac or VM), the script falls back to CPU automatically.

---

## Troubleshooting

**`python3.11` not found:**
```bash
brew install python@3.11
```

**Camera permission denied:**
Go to System Settings → Privacy & Security → Camera and grant access to Terminal (or your IDE).

**Low FPS on Intel Mac:**
MPS requires Apple Silicon. On Intel, YOLO will run on CPU, which is slower. Consider using a smaller model: `--model yolov8n-pose.pt`.

**iVCam not detected:**
Ensure the iVCam app is open on both your iPhone and Mac before launching the script.
