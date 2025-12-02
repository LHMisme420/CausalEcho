# webcam_realtime.py
import cv2
from causal_echo_detector import CausalEchoDetector

detector = CausalEchoDetector(threshold=0.7)
cap = cv2.VideoCapture(0)

print("CausalEcho Live — Look into the camera. Are you real?")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    score = detector.analyze_frame(frame)
    label = "REAL" if score >= 0.7 else "DEEPFAKE"
    color = (0, 255, 0) if score >= 0.7 else (0, 0, 255)

    cv2.putText(frame, f"{label} ({score:.3f})", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("CausalEcho — Reality Mirror", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
