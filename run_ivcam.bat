@echo off
title Enhanced Tracking - iVCam
echo Starting with iVCam (phone camera)...
call C:\Users\mohd1\OneDrive\Desktop\yolo_env311\Scripts\activate.bat
cd /d C:\Users\mohd1\OneDrive\Desktop\_ziptest
python person_hand_finger_both_ivcam_enhanced.py --index 0 --backend dshow --model yolov8n.pt --width 1920 --height 1080 --fps 60
pause
