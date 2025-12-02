# causal_echo_detector.py
import os
import cv2
import numpy as np
import hashlib
from scipy.signal import correlate
import requests
from tqdm import tqdm
import argparse
from utils.physics_oracles import query_light_coherence, query_environmental_oracle
from utils.fingerprint import extract_sensor_fingerprint

class CausalEchoDetector:
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def analyze_frame(self, frame):
        scores = []

        # 1. Light & Shadow Physics
        light_score = query_light_coherence(frame)
        scores.append(light_score)

        # 2. Sensor Fingerprint (real cameras have unique noise)
        fp = extract_sensor_fingerprint(frame)
        fingerprint_score = 0.9 if len(set(fp)) > 100 else 0.1
        scores.append(fingerprint_score)

        # 3. Environmental Causality (mock — real: GPS + time → weather/seismic)
        env_score = query_environmental_oracle()
        scores.append(env_score)

        return np.mean(scores)

    def scan_file(self, filepath):
        print(f"Scanning: {os.path.basename(filepath)}")
        
        frames = []
        if filepath.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img = cv2.imread(filepath)
            if img is not None:
                frames = [img]
        else:
            cap = cv2.VideoCapture(filepath)
            while len(frames) < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

        if not frames:
            print("Could not read file.")
            return

        causality_scores = []
        for frame in tqdm(frames, desc="Verifying physics", leave=False):
            score = self.analyze_frame(frame)
            causality_scores.append(score)

        avg_causality = np.mean(causality_scores)
        label = "REAL ✔" if avg_causality >= self.threshold else "DEEPFAKE ✘ (Acausal)"
        confidence = avg_causality * 100

        print(f"Result: [{label}] {confidence:6.2f}% causal alignment")
        return avg_causality

    def scan_folder(self, folder_path):
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.jpg', '.png', '.webp'))]
        
        results = []
        for f in tqdm(files, desc="Batch scanning"):
            score = self.scan_file(f)
            results.append((f, score))

        fakes = sum(1 for _, s in results if s < self.threshold)
        print(f"\nFinal Verdict: {fakes}/{len(results)} files are ACAUSAL (deepfakes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CausalEcho — Physics-Based Deepfake Disprover")
    parser.add_argument("path", help="File or folder to scan")
    parser.add_argument("--threshold", type=float, default=0.75, help="Causality threshold (0.0–1.0)")
    args = parser.parse_args()

    detector = CausalEchoDetector(args.threshold)

    if os.path.isfile(args.path):
        detector.scan_file(args.path)
    elif os.path.isdir(args.path):
        detector.scan_folder(args.path)
    else:
        print("Path not found!")
