@echo off
cd /d C:\Users\mohd1\OneDrive\Desktop\_ziptest
echo ============================================
echo   Enhanced Tracking v4.0
echo ============================================
echo.
call yolo_env311\Scripts\activate.bat
if exist "yolov8s-pose.pt" (set MODEL=yolov8s-pose.pt) else (set MODEL=yolov8n.pt)
echo.
echo  [1] Laptop Camera   (720p 30fps)
echo  [2] DroidCam USB    (720p 60fps)
echo.
set /p choice="Select camera (1 or 2): "
if "%choice%"=="1" (
    python person_hand_finger_both_ivcam_enhanced.py --index 3 --backend dshow --width 1280 --height 720 --fps 30 --model %MODEL%
) else if "%choice%"=="2" (
    python person_hand_finger_both_ivcam_enhanced.py --index 0 --backend msmf --width 1280 --height 720 --fps 60 --model %MODEL%
)
pause
