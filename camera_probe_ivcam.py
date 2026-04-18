import cv2
import numpy as np

def probe(max_index=8):
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("ANY",  cv2.CAP_ANY),
    ]

    for idx in range(max_index):
        for name, be in backends:
            cap = cv2.VideoCapture(idx, be)
            if not cap.isOpened():
                continue

            # لا تفرض دقة كبيرة بالبداية
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # warmup
            frame = None
            ok = False
            for _ in range(30):
                ok, frame = cap.read()ئئ
                if ok and frame is not None:
                    pass

            if ok and frame is not None:
                mean = float(frame.mean())
                out = f"probe_idx{idx}_{name}.jpg"
                cv2.imwrite(out, frame)
                print(f"[{idx} {name}] read={ok} shape={frame.shape} mean={mean:.1f} -> {out}")
            cap.release()

if __name__ == "__main__":
    probe(10)
