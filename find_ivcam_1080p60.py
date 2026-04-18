import cv2
import time

BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF",  cv2.CAP_MSMF),
    ("ANY",   cv2.CAP_ANY),
]

def try_cam(index, backend_name, backend_id, w=1920, h=1080, fps=60):
    cap = cv2.VideoCapture(index, backend_id)
    if not cap.isOpened():
        return None

    # اطلب 1080p + 60fps + MJPG (غالبًا أفضل مع iVCam)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # warmup
    frame = None
    ok = False
    for _ in range(40):
        ok, frame = cap.read()

    if not ok or frame is None:
        cap.release()
        return None

    mean = frame.mean()
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    # احفظ لقطة للتأكد مش سوداء
    out = f"ivcam_idx{index}_{backend_name}.jpg"
    cv2.imwrite(out, frame)

    return {
        "index": index,
        "backend": backend_name,
        "mean": mean,
        "frame_shape": frame.shape,
        "reported_w": actual_w,
        "reported_h": actual_h,
        "reported_fps": actual_fps,
        "snapshot": out,
    }

def main():
    print("Scanning cameras for iVCam... (0..12)")
    best = None

    for idx in range(13):
        for name, be in BACKENDS:
            r = try_cam(idx, name, be, 1920, 1080, 60)
            if r:
                print(f"[OK] idx={idx} backend={name} shape={r['frame_shape']} mean={r['mean']:.1f} "
                      f"reported={r['reported_w']}x{r['reported_h']} fps={r['reported_fps']:.1f} -> {r['snapshot']}")

                # اعتبره صالح إذا مش أسود
                if r["mean"] > 5.0 and best is None:
                    best = r

    if best:
        print("\n✅ BEST FOUND (first non-black):")
        print(f"Use: --index {best['index']} --backend {best['backend'].lower()} --width 1920 --height 1080")
        print(f"Snapshot: {best['snapshot']}")
    else:
        print("\n❌ No camera opened with OpenCV.")
        print("If iVCam works in Windows Camera but not here:")
        print("- Close Windows Camera completely")
        print("- Reinstall iVCam PC driver (virtual camera)")
        print("- Try running PowerShell as Administrator once")
        print("- Make sure iVCam is in 'Virtual Camera' mode / enabled")

if __name__ == "__main__":
    main()
