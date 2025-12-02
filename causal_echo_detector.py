# causal_echo_detector.py
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from utils.physics_oracles import query_light_coherence, query_environmental_oracle
from utils.fingerprint import extract_sensor_fingerprint

class CausalEchoDetector:
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def analyze_frame(self, frame):
        scores = []
        scores.append(query_light_coherence(frame))
        fp = extract_sensor_fingerprint(frame)
        scores.append(0.96 if len(np.unique(fp)) > 800 else 0.15)
        scores.append(query_environmental_oracle())
        return np.mean(scores)

    def scan_file(self, filepath):
        print(f"\nScanning → {os.path.basename(filepath)}")
        frames = []
        if filepath.lower().endswith(('.jpg','.jpeg','.png','.webp')):
            img = cv2.imread(filepath)
            if img is not None: frames = [img]
        else:
            cap = cv2.VideoCapture(filepath)
            while len(frames) < 32:
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
            cap.release()

        if not frames:
            print("Error reading file.")
            return

        scores = [self.analyze_frame(f) for f in tqdm(frames, desc="Consulting physics", leave=False)]
        avg = np.mean(scores)
        verdict = "REAL" if avg >= self.threshold else "DEEPFAKE (Acausal)"
        color = "\033[92m" if avg >= self.threshold else "\033[91m"
        print(f"{color}VERDICT: {verdict} → {avg*100:.2f}% causal alignment\033[0m")
        return avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CausalEcho — Physics vs Deepfakes")
    parser.add_argument("path", help="File or folder path")
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    detector = CausalEchoDetector(args.threshold)
    if os.path.isfile(args.path):
        detector.scan_file(args.path)
    elif os.path.isdir(args.path):
        for f in os.listdir(args.path):
            if f.lower().endswith(('.mp4','.mov','.avi','.jpg','.png','.webp')):
                detector.scan_file(os.path.join(args.path, f))
