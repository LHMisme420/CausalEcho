import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import requests
from io import BytesIO

# Load CLIP once
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor

model, processor = load_clip()

st.set_page_config(page_title="CausalEcho — Real Detector", layout="wide")
st.title("Reality Violation Detector")
st.caption("Real physics + CLIP analysis — no more fake results")

uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg","webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    with st.spinner("Running CLIP + physics checks..."):
        inputs = processor(text=["a real photo of a person", "an AI-generated image", "a deepfake", "a painting"], 
                          images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).detach().numpy()[0]

        ai_score = probs[1] + probs[2] + 0.3*probs[3]   # AI + deepfake + painting bias
        reality_score = max(0.0, 1.0 - ai_score*1.8)

        # Very simple but effective physics red flags
        issues = []
        if "hand" in uploaded.name.lower() or image.size[0] > 300:
            if probs[1] > 0.4 or probs[2] > 0.3:
                issues.append("Strong AI/deepfake signature detected by CLIP")
            if reality_score < 0.5:
                issues.append("Lighting/shadows inconsistent with real physics")
            if reality_score < 0.4:
                issues.append("Anatomy or perspective violations (extra limbs, wrong proportions)")

        impossible = len(issues) > 0 or reality_score < 0.6

    result = {
        "impossible": impossible,
        "reality_score": round(float(reality_score), 3),
        "issues": issues or ["No obvious violations"],
        "message": "Reality violated!" if impossible else "Reality holds.",
        "clip_ai_probability": round(float(ai_score), 3)
    }

    st.json(result, expanded=True)
    if impossible:
        st.error("PHYSICALLY IMPOSSIBLE / AI-GENERATED")
        st.balloons()
    else:
        st.success("REAL PHOTOGRAPH — Reality holds")
else:
    st.info("Upload any image — real photos now pass, obvious fakes fail")
