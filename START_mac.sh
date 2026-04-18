#!/usr/bin/env bash
# =====================================================================
#  START_mac.sh — Launch Enhanced CV Tracking on macOS
# =====================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source yolo_env311/bin/activate

echo ""
echo "============================="
echo "  Enhanced CV Tracking (Mac)"
echo "============================="
echo ""
echo "  1) iVCam camera     (index 0, 1920x1080)"
echo "  2) Built-in camera  (index 0 or 1, 1280x720)"
echo ""
read -rp "Select mode [1/2]: " MODE

case "$MODE" in
  1)
    echo "Starting with iVCam (index 0, 1920x1080)..."
    python person_hand_finger_mac.py \
      --index 0 \
      --backend any \
      --width 1920 \
      --height 1080 \
      --model yolov8s-pose.pt
    ;;
  2)
    echo "Probing built-in camera index..."
    # Try index 1 first (common on MacBooks with iVCam also connected), then 0
    python - <<'EOF'
import cv2
for idx in range(4):
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 5:
            print(f"  Found working camera at index {idx} ({frame.shape[1]}x{frame.shape[0]})")
        cap.release()
EOF
    read -rp "Enter camera index to use [default 0]: " CAM_IDX
    CAM_IDX="${CAM_IDX:-0}"
    echo "Starting with built-in camera (index ${CAM_IDX}, 1280x720)..."
    python person_hand_finger_mac.py \
      --index "$CAM_IDX" \
      --backend any \
      --width 1280 \
      --height 720 \
      --model yolov8s-pose.pt
    ;;
  *)
    echo "Invalid selection. Exiting."
    exit 1
    ;;
esac
