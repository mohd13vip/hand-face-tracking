from ultralytics import YOLO
import cv2
import time
import torch

MODEL_PATH = r"Task1_ChallengeB\models\yolov8s_best.pt"  # you can change to yolov8n_best.pt or yolov8m_best.pt

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows-friendly
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    fps_hist = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = model(frame, conf=0.25, device=device, verbose=False)
        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_hist.append(fps)
        avg_fps = sum(fps_hist[-30:]) / len(fps_hist[-30:])

        annotated = results[0].plot()

        cv2.putText(
            annotated, f"FPS: {avg_fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        cv2.imshow("Webcam YOLO Demo (press q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
