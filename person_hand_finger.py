import time
import cv2
from ultralytics import YOLO

import mediapipe as mp
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


def main():
    # ---- Camera (your working one) ----
    CAM_INDEX = 1
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Camera not opened. Try index 0/1/2.")
        return

    # ---- YOLO person ----
    yolo = YOLO("yolov8n.pt")
    PERSON_CLASS = 0

    # ---- MediaPipe HandLandmarker (VIDEO mode for better tracking) ----
    model_path = "hand_landmarker.task"
    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    # previous fingertip for each detected hand (0 and 1)
    prev_index_tip = [None, None]

    last_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Failed to read frame")
            break

        # -------- Person detection --------
        yres = yolo.predict(frame, conf=0.20, classes=[PERSON_CLASS], verbose=False)
        for box in yres[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "person", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # -------- Hand detection (VIDEO mode) --------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks[:2]):
                # different color per hand
                color = (0, 255, 0) if i == 0 else (0, 200, 255)

                pts = draw_hand(frame, hand_lms, color=color)

                # fingertip index 8
                index_tip = pts[8]
                cv2.circle(frame, index_tip, 8, (0, 0, 255), -1)

                # movement
                if prev_index_tip[i] is not None:
                    cv2.arrowedLine(frame, prev_index_tip[i], index_tip, (0, 0, 255), 3)
                    dx = index_tip[0] - prev_index_tip[i][0]
                    dy = index_tip[1] - prev_index_tip[i][1]
                    cv2.putText(frame, f"hand{i} dx={dx} dy={dy}",
                                (20, 60 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                prev_index_tip[i] = index_tip

                # handedness label (Left/Right)
                if result.handedness and i < len(result.handedness):
                    label = result.handedness[i][0].category_name
                    score = result.handedness[i][0].score
                    cv2.putText(frame, f"{label} ({score:.2f})",
                                (20, 120 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # -------- FPS --------
        now = time.time()
        fps = 1.0 / max(1e-6, (now - last_t))
        last_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Person + TWO Hands + Fingers (press q)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
