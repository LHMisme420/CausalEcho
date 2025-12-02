# utils/fingerprint.py
import numpy as np

def extract_sensor_fingerprint(frame):
    noise = frame.astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (21,21), 12)
    noise = noise - blurred
    return noise.flatten()
