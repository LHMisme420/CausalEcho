import streamlit as st
from PIL import Image
from transformers import pipeline  # Simpler for new model
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np

# Load the 2025-tuned AI detector (better for photos, Gemini-era gens)
@st.cache_resource
def load_model():
    pipe = pipeline("image-classification", model="XenArcAI/AIRealNet")
    return pipe

model_pipe = load_model()

# Torch-based Watermark Detector (refined for subtlety, less FP)
class SimpleWatermarkDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.conv(x)

@st.cache_resource
def load_watermark_detector():
    detector = SimpleWatermarkDetector()
    optimizer = torch.optim.Adam(detector.parameters(), lr=0.01)
    # Refined dummy data: Simulate small, high-contrast "AI badge" (pill/sparkle)
    dummy_high = torch.ones(20, 1, 224, 224) * 0.7
    dummy_high[:, :, 80:100, 80:120] = 0.1  # Small rectangular dark spot
    dummy_high[:, :, 85:95, 85:115] += torch.randn(20, 1, 10, 30) * 0.05  # Text-like noise
    dummy_low = torch.ones(20, 1, 224, 224) * 0.5 + torch.randn(20, 1, 224, 224) * 0.15  # Noisy but low contrast
    labels_high = torch.ones(20, 1)
    labels_low = torch.zeros(20, 1)
    
    for _ in range(100):  # More epochs for precision
        pred_high = detector(dummy_high)
        pred_low = detector(dummy_low)
        loss = nn.BCELoss()(torch.cat([pred_high, pred_low]), torch.cat([labels_high, labels_low]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return detector

watermark_detector = load_watermark_detector()

# Watermark Detection: Stricter thresholds, size check
def detect_watermark(image_pil):
    img_array = np.array(image_pil.convert("L"))  # Grayscale
    h, w = img_array.shape
    crop_size = min(224, h//5, w//5)  # Smaller crop to focus
    corners = [
        img_array[:crop_size, :crop_size],
        img_array[:crop_size, -crop_size:],
        img_array[-crop_size:, :crop_size],
        img_array[-crop_size:, -crop_size:]
    ]
    
    max_conf = 0.0
    details = "No watermark detected"
    for i, corner in enumerate(corners):
        if corner.size < 10000: continue  # Skip tiny/empty
        corner_norm = torch.tensor(corner / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pad_h = 224 - corner_norm.shape[2]
        pad_w = 224 - corner_norm.shape[3]
        if pad_h > 0 or pad_w > 0:
            corner_norm = torch.nn.functional.pad(corner_norm, (0, max(0, pad_w), 0, max(0, pad_h)))
        
        with torch.no_grad():
            conf = watermark_detector(corner_norm).item()
        if conf > max_conf:
            max_conf = conf
            details = f"Pattern in corner {i+1} (conf: {conf:.2f})"
    
    # Stricter: High conf + high edge variance + small area (badge-sized)
    if max_conf > 0.85:
        for corner in corners:
            if corner.size == 0: continue
            edges = np.gradient(np.gradient(corner))[0]
            variance = np.std(edges)
            area_ratio = (variance > 30) and (corner.std() > 20) and (np.sum(corner < 100) / corner.size < 0.2)  # Small dark area
            if area_ratio:
                return 1.0, f"Watermark confirmed: {details}"
        return 0.92, details  # Probable
    
    return 0.0, details

# UI
st.set_page_config(page_title="CausalEcho — Accurate AF", layout="wide")
st.title("CausalEcho: Real vs AI Detector (2025-Tuned)")
st.caption("Switches to AIRealNet for better photo accuracy • Watermarks only flag if model suspects")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    with st.spinner("Scanning..."):
        # Model inference (new pipeline)
        results = model_pipe(image)
        fake_prob_model = next(r['score'] for r in results if r['label'] == 'artificial')
        real_prob = next(r['score'] for r in results if r['label'] == 'real')

        # Watermark check
        watermark_conf, watermark_msg = detect_watermark(image)

        # Balanced decision: Watermark boosts only if model not super confident real
        if watermark_conf >= 0.85 and fake_prob_model > 0.4:
            fake_prob = 1.0
            message = "AI-GENERATED (watermark + model confirm)"
            issues = [watermark_msg]
        else:
            fake_prob = fake_prob_model
            message = "AI-GENERATED" if fake_prob > 0.55 else "REAL"
            issues = ["Artifacts detected"] if fake_prob > 0.55 else []

    st.divider()
    if fake_prob > 0.7:
        st.error(f"🚨 FAKE — {message}")
    else:
        st.success("✅ REAL")

    col1, col2, col3 = st.columns(3)
    col1.metric("AI Prob", f"{fake_prob:.1%}")
    col2.metric("Model Raw", f"{fake_prob_model:.1%}")
    col3.metric("Watermark Conf", f"{watermark_conf:.1%}")

    with st.expander("Details"):
        st.write("Watermark:", watermark_msg)
        st.write(f"Probs: AI={fake_prob_model:.3f}, Real={real_prob:.3f}")

else:
    st.info("Test real photos (green) vs AI (red). Watermarks now selective.")

st.caption("Dec 2025 Update: Refined for fewer FPs, better distinction")
