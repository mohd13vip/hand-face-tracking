@echo off
echo Downloading YOLOv8s-Pose model...
call yolo_env311\Scripts\activate.bat
python -c "from ultralytics import YOLO; m = YOLO('yolov8s-pose.pt'); print('Model downloaded OK:', m.model_name)"
echo.
echo Done! You can now run START.bat
pause
