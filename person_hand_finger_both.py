import time
import cv2
import mediapipe as mp
from ultralytics import YOLO

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def draw_hand(frame, landmarks, color):
    h, w = frame.shape[:2]
    pts = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))
        cv2.circle(frame, (x, y), 3, color, -1)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)

    return pts

def label_from_handedness(result, i):
    # result.handedness: list[list[Category]]
    try:
        return result.handedness[i][0].category_name  # "Left" or "Right"
    except Exception:
        return f"Hand{i}"

def open_camera(cam_index: int):
    # Try DSHOW first (usually best on Windows), then MSMF
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # warmup
    for _ in range(20):
        cap.read()

    if cap.isOpened():
        return cap, "DSHOW"

    cap.release()

    cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for _ in range(20):
        cap.read()

    if cap.isOpened():
        return cap, "MSMF"

    cap.release()
    return None, None

def main():
    # ✅ غيّر هذا الرقم حسب camera_probe.py
    cam_index = 2

    cap, backend = open_camera(cam_index)
    if cap is None:
        print("❌ Camera not opened.")
        print("✅ Run: python camera_probe.py and change cam_index.")
        return

    print(f"✅ Camera opened: index={cam_index}, backend={backend}")

    # YOLO person detection
    yolo = YOLO("yolov8n.pt")
    person_class = 0

    # MediaPipe HandLandmarker (Tasks API)
    model_path = "hand_landmarker.task"
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    prev_tip = {}  # key: "Left"/"Right"/"Hand0"... value: (x,y)
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("❌ Failed to read frame (camera might be busy).")
            break

        # --- YOLO person detection ---
        yres = yolo.predict(frame, conf=0.25, classes=[person_class], verbose=False)
        for box in yres[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "person", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # --- MediaPipe hands ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for i, lms in enumerate(result.hand_landmarks):
                label = label_from_handedness(result, i)

                # color by hand label
                color = (0, 255, 0) if label == "Left" else (0, 255, 255)

                pts = draw_hand(frame, lms, color=color)

                # Index fingertip landmark = 8
                tip = pts[8]
                cv2.circle(frame, tip, 8, (0, 0, 255), -1)
                cv2.putText(frame, label, (tip[0] + 10, tip[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # movement
                prev = prev_tip.get(label, None)
                if prev is not None:
                    cv2.arrowedLine(frame, prev, tip, (0, 0, 255), 2, tipLength=0.25)
                    dx = tip[0] - prev[0]
                    dy = tip[1] - prev[1]
                    y_text = 60 + 25 * i
                    cv2.putText(frame, f"{label} index dx={dx} dy={dy}", (20, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                prev_tip[label] = tip

        # FPS
        now = time.time()
        fps = 1.0 / max(1e-6, (now - t_prev))
        t_prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Person + BOTH Hands + Index Movement (press q)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
