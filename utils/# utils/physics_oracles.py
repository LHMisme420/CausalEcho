# utils/physics_oracles.py
import numpy as np
import cv2

def query_light_coherence(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    coherence = np.std(edges)
    return min(coherence / 50.0, 1.0)  # Higher variance = more natural

def query_environmental_oracle():
    # In real app: query weather, seismic, GPS, satellite, etc.
    # For now: simulate plausible causality
    return np.random.uniform(0.7, 0.98)
