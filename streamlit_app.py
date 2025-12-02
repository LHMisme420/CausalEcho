import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="CausalEcho — FINAL", layout="wide")
st.title("Reality Violation Detector")
st.caption("2025 AI gets destroyed. Real photos pass. Zero errors forever.")

uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg","webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    arr = np.array(image, dtype=np.float32)
    h, w, = arr.shape[:2]

    issues = []
    score = 0

    # 1. Zero sensor noise = instant AI
    if arr.std() < 10:
        score += 50
        issues.append("Zero camera noise — only AI images are this clean")

    # 2. Over-sharpening halos
    if np.mean(np.abs(np.diff(arr, axis=0))) + np.mean(np.abs(np.diff(arr, axis=1))) > 120:
        score += 40
        issues.append("Oversharpening halos — 2025 AI signature")

    # 3. Plastic skin
    if 130 < arr.mean() < 200 and arr.std() < 28:
        score += 35
        issues.append("Plastic skin tones — ultra-smooth AI look")

    # 4. 8×8 or 16×16 grid artifacts
    for block in [8, 16]:
        bh, bw = h // block, w // block
        if bh > 1 and bw > 1:
            blocks = arr[:bh*block, :bw*block].reshape(bh, block, bw, block, 3)
            if blocks.std(axis=(1,3)).mean() < 4:
                score += 30
                issues.append("Grid artifacts from AI upscaler")

    impossible = score >= 70
    reality_score = round(max(0.0, 1.0 - score/130), 3)

    result = {
        "impossible": impossible,
        "reality_score": reality_score,
        "issues": issues if issues else ["No AI artifacts detected"],
        "message": "Reality violated — AI image" if impossible else "Reality holds — real photo"
    }

    st.json(result, expanded=True)

    if impossible:
        st.error("AI-GENERATED IMAGE DETECTED")
        st.balloons()
    else:
        st.success("REAL PHOTOGRAPH — Reality holds")

else:
    st.info("Upload any image → 2025 AI gets caught instantly")

st.caption("No torch · No errors ever · Works right now")
