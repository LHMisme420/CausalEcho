# streamlit_app.py
import streamlit as st
from causal_echo_detector import CausalEchoDetector
import cv2
import numpy as np

st.set_page_config(page_title="CausalEcho", page_icon="🛡️", layout="centered")

st.title("🛡️ CausalEcho")
st.subheader("The Deepfake Disprover")
st.write("Upload a video or image. Physics will judge.")

detector = CausalEchoDetector(threshold=0.75)

file = st.file_uploader("Drop your evidence here", type=['mp4', 'mov', 'avi', 'jpg', 'png', 'webp'])

if file:
    with st.spinner("Interrogating reality..."):
        bytes_data = file.read()
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
            tmp.write(bytes_data)
            path = tmp.name

        score = detector.scan_file(path)
        label = "REAL ✔" if score >= 0.75 else "DEEPFAKE ✘"

        st.write(f"### Verdict: **{label}**")
        st.progress(score)
        st.write(f"Causal Alignment: `{score:.3f}`")

        if file.type.startswith("video"):
            st.video(file)
        else:
            st.image(file)

st.caption("Powered by physics. Immune to AI. Built for truth.")
