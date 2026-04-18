"""Quick camera index finder - run this to see which indices work."""
import cv2
print("Scanning camera indices 0-9...\n")
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print("  Index {}: WORKS  ({}x{})".format(i, w, h))
        else:
            print("  Index {}: Opens but no frame".format(i))
        cap.release()
    else:
        print("  Index {}: Not available".format(i))
print("\nDone. Use the working index with: --index <number>")
input("Press Enter to close...")
