import time
import argparse
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
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], color, 2)

    return pts


def label_from_handedness(result, i):
    try:
        return result.handedness[i][0].category_name  # "Left" / "Right"
    except Exception:
        return f"Hand{i}"


def backend_to_cv(backend_name: str) -> int:
    backend_name = backend_name.lower()
    if backend_name == "dshow":
        return cv2.CAP_DSHOW
    if backend_name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Camera index (iVCam often 0..5)")
    parser.add_argument("--backend", type=str, default="dshow", choices=["dshow", "msmf", "any"], help="OpenCV backend")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence for person")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--hand_model", type=str, default="hand_landmarker.task", help="MediaPipe task file")
    parser.add_argument("--no_set_props", action="store_true", help="Don't force width/height/fps (use camera defaults)")
    args = parser.parse_args()

    cap_backend = backend_to_cv(args.backend)

    # ---- open camera ----
    cap = cv2.VideoCapture(args.index, cap_backend)

    if not cap.isOpened():
        print("❌ Camera not opened. Try another --index or --backend (msmf/dshow).")
        return

    # ---- try set camera properties ----
    if not args.no_set_props:
        # MJPG helps iVCam keep higher fps sometimes
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
        cap.set(cv2.CAP_PROP_FPS, int(args.fps))

    # warmup frames
    for _ in range(20):
        cap.read()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ Opened camera index={args.index} backend={args.backend}")
    print(f"   Requested: {args.width}x{args.height} @ {args.fps}fps")
    print(f"   Actual:    {actual_w}x{actual_h} @ {actual_fps}fps")

    # ---- YOLO for person ----
    yolo = YOLO(args.model)
    person_class = 0

    # ---- MediaPipe HandLandmarker (Tasks API) ----
    base_options = python.BaseOptions(model_asset_path=args.hand_model)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    prev_tip = {}  # label -> (x,y)
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("❌ Failed to read frame")
            break

        # ---- YOLO person ----
        yres = yolo.predict(frame, conf=args.conf, classes=[person_class], verbose=False)
        if len(yres) > 0 and yres[0].boxes is not None:
            for box in yres[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "person", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # ---- MediaPipe hands ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for i, lms in enumerate(result.hand_landmarks):
                label = label_from_handedness(result, i)
                color = (0, 255, 0) if label == "Left" else (0, 255, 255)

                pts = draw_hand(frame, lms, color=color)
                if len(pts) < 9:
                    continue

                # index fingertip = 8
                tip = pts[8]
                cv2.circle(frame, tip, 8, (0, 0, 255), -1)
                cv2.putText(frame, label, (tip[0] + 10, tip[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                prev = prev_tip.get(label)
                if prev is not None:
                    cv2.arrowedLine(frame, prev, tip, (0, 0, 255), 2, tipLength=0.25)
                    dx = tip[0] - prev[0]
                    dy = tip[1] - prev[1]
                    y_text = 60 + 25 * i
                    cv2.putText(frame, f"{label} index dx={dx} dy={dy}", (20, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                prev_tip[label] = tip

        # ---- FPS ----
        now = time.time()
        fps = 1.0 / max(1e-6, (now - t_prev))
        t_prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Person + BOTH Hands (press q)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
