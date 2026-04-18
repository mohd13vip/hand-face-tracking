Setup Guide

1) Create venv:
python -m venv yolo_env
yolo_env\Scripts\activate

2) Install:
pip install -r requirements.txt

3) Verify:
python verify_setup.py


# Setup Guide (Task 1 + Task 2)

## System Requirements
- Python **3.10 or 3.11**
- NVIDIA GPU with CUDA (recommended) for training/inference
- ~20GB free storage (more if you keep many MOT videos)

## Installation
```bash
python -m venv yolo_env
# Windows
yolo_env\Scripts\activate
pip install -r requirements.txt
```

## Dataset Setup
### Pascal VOC 2012 (Task 1)
Expected folder (example):
```
datasets/
└── VOCdevkit/
    └── VOC2012/
        ├── Annotations/
        ├── JPEGImages/
        └── ImageSets/
```
Then run your conversion script (example):
```bash
python convert_voc_to_yolo.py
```

### MOT17 (Task 2)
Expected folder:
```
datasets/
└── MOT17/
    └── train/
        ├── MOT17-02-DPM/
        ├── MOT17-04-DPM/
        └── ...
```

## Verify Setup
```bash
python verify_setup.py
```
You should see confirmations for Python version, PyTorch, and key packages.
