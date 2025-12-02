# utils/fingerprint.py
import numpy as np

def extract_sensor_fingerprint(frame):
    noise = frame.astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (21,21), 12)
    noise = noise - blurred
    return noise.flatten()
# Temporary stub so the app runs while you finish the real one
def extract_sensor_fingerprint(frame):
    return {
        "sensor_dust_score": 0.0,
        "prnu_noise_consistency": 1.0,
        "fingerprint_match": True
    }
