#!/usr/bin/env bash
# =====================================================================
#  run_builtin_cam_mac.sh — Run with Mac built-in camera at 1280x720
# =====================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source yolo_env311/bin/activate

# Default to index 0; change to 1 if iVCam occupies index 0
CAM_IDX="${1:-0}"

echo "Starting Enhanced CV Tracking — built-in camera (index ${CAM_IDX}, 1280x720)..."
echo "Tip: pass a different index as argument, e.g.:  ./run_builtin_cam_mac.sh 1"
python person_hand_finger_mac.py \
  --index "$CAM_IDX" \
  --backend any \
  --width 1280 \
  --height 720 \
  --model yolov8s-pose.pt
