import streamlit as st
from causal_echo_detector import CausalEchoDetector
import os
import cv2
import numpy as np

st.set_page_config(page_title="CausalEcho", page_icon="🌌", layout="centered")
st.title("🌌 CausalEcho")
st.markdown("**Deepfakes don’t lie — they break physics.**  \nUpload anything. If reality breaks, we catch it.")

uploaded = st.file_uploader("Drop video/image", type=["mp4","mov","avi","jpg","png","jpeg","webm","gif"])

if uploaded:
    if uploaded.type.startswith("video"):
        st.video(uploaded)
    else:
        st.image(uploaded)
    
    with st.spinner("Enforcing physics, gravity, light, causality..."):
        temp_path = "temp_upload"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        
        detector = CausalEchoDetector()
        result = detector.analyze(temp_path)  # ← REAL checks now
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if result.get("impossible", False):
            st.error("PHYSICALLY IMPOSSIBLE → DEEPFAKE CONFIRMED")
            st.json(result, expanded=True)
        else:
            st.success("Reality holds. No violations detected.")
            st.json(result, expanded=True)
            st.balloons()