#!/usr/bin/env bash
# =====================================================================
#  run_ivcam_mac.sh — Run with iVCam at 1920x1080
# =====================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source yolo_env311/bin/activate

echo "Starting Enhanced CV Tracking — iVCam (index 0, 1920x1080)..."
python person_hand_finger_mac.py \
  --index 0 \
  --backend any \
  --width 1920 \
  --height 1080 \
  --model yolov8s-pose.pt
