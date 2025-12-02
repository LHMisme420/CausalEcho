# utils/physics_oracles.py
import cv2
import numpy as np

def query_light_coherence(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return min(np.std(edges) / 42.0, 1.0)

def query_environmental_oracle():
    # Future: connect to USGS, NOAA, satellite data
    return np.random.uniform(0.73, 0.99)
