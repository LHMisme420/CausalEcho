# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import tempfile
from causal_echo_detector import CausalEchoDetector

st.set_page_config(page_title="CausalEcho", page_icon="shield")
st.title("CausalEcho")
st.caption("The first physics-based deepfake disprover")

detector = CausalEchoDetector(threshold=0.75)
file = st.file_uploader("Drop video/image here", type=['mp4','mov','avi','jpg','png','webp'])

if file:
    with st.spinner("Interrogating the laws of physics..."):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1])
        tfile.write(file.read())
        tfile.close()
        score = detector.scan_file(tfile.name)
        os.unlink(tfile.name)

        verdict = "REAL" if score >= 0.75 else "DEEPFAKE (Acausal)"
        st.metric("Verdict", verdict, f"{score*100:.1f}%")
        st.progress(score)
        if file.type.startswith("video"):
            st.video(file)
        else:
            st.image(file)
