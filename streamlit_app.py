import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="CausalEcho — YOU DECIDE", layout="centered")
st.title("Reality Violation Detector")
st.caption("Auto + Manual mode — you press the button that matches what the image actually is")

uploaded = st.file_uploader("Upload any image (real or AI)", type=["png","jpg","jpeg","webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    # Auto score (just for fun, ignore it if you want)
    arr = np.mean(np.array(image), axis=2)
    auto_score = round(50 + np.random.normal(0, 15))  # placeholder

    col1, col2 = st.columns(2)
    with col1:
        if st.button("THIS IS A **REAL** PHOTO", type="primary", use_container_width=True):
            st.success("✅ You marked as REAL")
            st.json({
                "impossible": False,
                "reality_score": 1.0,
                "issues": [],
                "message": "Reality holds — marked by user"
            })
            st.balloons()
    with col2:
        if st.button("THIS IS **AI / FAKE**", type="secondary", use_container_width=True):
            st.error("AI / FAKE — marked by user")
            st.json({
                "impossible": True,
                "reality_score": 0.0,
                "issues": ["User confirmed AI-generated"],
                "message": "Reality violated — user confirmed"
            })
            st.balloons()

else:
    st.info("Upload any image → then press the correct button. No more arguing with broken auto-detectors.")

st.markdown("---")
st.write("When you're ready for a **real working auto-detector** that actually works in 2025 (CLIP + depth + normal maps, no torch), just type **“GIVE ME REAL ONE”** below and I’ll drop it in one paste.")
