import cv2
import time

BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF",  cv2.CAP_MSMF),
    ("ANY",   cv2.CAP_ANY),
]

def try_cam(index, api_name, api):
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        print(f"[{index} {api_name}] open=False")
        return False

    # Common Windows fixes for black frames:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    time.sleep(0.2)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print(f"[{index} {api_name}] read=False")
        return False

    mean_val = float(frame.mean())
    print(f"[{index} {api_name}] read=True shape={frame.shape} mean={mean_val:.1f}")

    out = f"probe_idx{index}_{api_name}.jpg"
    cv2.imwrite(out, frame)
    print(f"   saved -> {out} (open it to confirm it’s not black)")

    cv2.imshow(f"idx{index}_{api_name}", frame)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return mean_val > 5  # if near 0, it's basically black

found = False
for idx in range(0, 6):
    for name, api in BACKENDS:
        if try_cam(idx, name, api):
            print(f"\n✅ Use this: index={idx}, backend={name}")
            found = True
            raise SystemExit

if not found:
    print("\n❌ All tested cameras returned black/empty frames.")
    print("Close Zoom/Teams/Camera app, re-check Windows Camera privacy for Desktop apps, then try again.")
