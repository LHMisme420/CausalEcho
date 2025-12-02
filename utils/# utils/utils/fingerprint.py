# utils/fingerprint.py
import numpy as np

def extract_sensor_fingerprint(frame):
    # Simulate real sensor noise pattern (PRNU)
    noise = frame.astype(np.float32) - cv2.GaussianBlur(frame, (5,5), 10)
    return noise.flatten()[:10000]
