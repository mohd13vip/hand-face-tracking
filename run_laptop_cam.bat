@echo off
title Enhanced Tracking - Laptop Camera
echo Starting with Laptop Camera...
call C:\Users\mohd1\OneDrive\Desktop\yolo_env311\Scripts\activate.bat
cd /d C:\Users\mohd1\OneDrive\Desktop\_ziptest
python person_hand_finger_both_ivcam_enhanced.py --index 1 --backend dshow --model yolov8n.pt --width 1280 --height 720 --fps 30
pause
