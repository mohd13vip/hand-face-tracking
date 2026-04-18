#!/usr/bin/env bash
# =====================================================================
#  setup_mac.sh — One-time setup for Enhanced CV Tracking on macOS
#  Requirements: Python 3.11 available as python3.11
# =====================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="yolo_env311"

echo "========================================"
echo "  Enhanced CV Tracking — Mac Setup"
echo "========================================"

# ── Create virtual environment ────────────────────────────────
if [ -d "$ENV_NAME" ]; then
    echo "[1/6] Virtual environment '$ENV_NAME' already exists — skipping creation."
else
    echo "[1/6] Creating Python 3.11 virtual environment: $ENV_NAME"
    python3.11 -m venv "$ENV_NAME"
fi

source "$ENV_NAME/bin/activate"
echo "      Activated: $(which python)  $(python --version)"

# ── Upgrade pip ───────────────────────────────────────────────
echo "[2/6] Upgrading pip..."
pip install --upgrade pip --quiet

# ── Install PyTorch (MPS-compatible, no CUDA index URL) ──────
echo "[3/6] Installing PyTorch + torchvision (MPS build)..."
pip install torch torchvision --quiet

# ── Install CV / ML dependencies ──────────────────────────────
echo "[4/6] Installing mediapipe, ultralytics, opencv, and utilities..."
pip install \
    mediapipe \
    ultralytics \
    opencv-python \
    numpy \
    scipy \
    matplotlib \
    pillow \
    sounddevice \
    psutil \
    --quiet

# ── Download YOLOv8s-Pose model ───────────────────────────────
echo "[5/6] Downloading yolov8s-pose.pt..."
if [ -f "yolov8s-pose.pt" ]; then
    echo "      yolov8s-pose.pt already present — skipping."
else
    python - <<'EOF'
from ultralytics import YOLO
print("      Fetching yolov8s-pose.pt via ultralytics...")
YOLO("yolov8s-pose.pt")   # auto-downloads on first load
print("      Done.")
EOF
fi

# ── Verify setup ──────────────────────────────────────────────
echo "[6/6] Running verify_setup.py..."
if [ -f "verify_setup.py" ]; then
    python verify_setup.py
else
    python - <<'EOF'
import torch, cv2, mediapipe, ultralytics, numpy, scipy
print("  torch      :", torch.__version__)
print("  MPS avail  :", torch.backends.mps.is_available())
print("  opencv     :", cv2.__version__)
print("  mediapipe  :", mediapipe.__version__)
print("  ultralytics:", ultralytics.__version__)
print("  numpy      :", numpy.__version__)
print("")
print("All packages OK.")
EOF
fi

echo ""
echo "========================================"
echo "  Setup complete!"
echo "  Run:  ./START_mac.sh   to launch"
echo "========================================"
