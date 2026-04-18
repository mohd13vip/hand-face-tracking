import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # IMPORTANT: CAP_DSHOW
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Couldn't read frame from camera")
        break

    cv2.imshow("Camera test (press q)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
