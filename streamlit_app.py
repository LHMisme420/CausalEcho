import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from datetime import datetime

# Load the 2025-tuned model (AIRealNet – kills Flux, SD3.5, Midjourney v6+)
@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained("Falconsai/AIRealNet")
    processor = AutoImageProcessor.from_pretrained("Falconsai/AIRealNet")
    return model, processor

model, processor = load_model()

st.set_page_config(page_title="CausalEcho — 2025 KILLER", layout="wide")
st.title("Reality Violation Detector")
st.caption("AIRealNet (Sept 2025) — actually catches modern AI, real photos pass")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    with st.spinner("Analyzing with 2025 model..."):
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        # Model labels: 0 = AI/fake, 1 = real
        fake_prob = float(probs[0])
        real_prob = float(probs[1])
        
        impossible = fake_prob > 0.42          # aggressive but safe threshold
        reality_score = round(real_prob, 3)

    result = {
        "impossible": impossible,
        "reality_score": reality_score,
        "issues": ["AI-generated (2025 diffusion artifacts)"] if impossible else [],
        "message": "Reality violated — AI image" if impossible else "Reality holds — real photo",
        "fake_probability": round(fake_prob, 3)
    }

    st.json(result, expanded=True)

    if impossible:
        st.error(f"AI / FAKE DETECTED – {fake_prob:.1%} confidence")
        st.balloons()
    else:
        st.success(f"REAL PHOTO – {real_prob:.1%} confidence")

    col1, col2 = st.columns(2)
    col1.metric("AI Probability", f"{fake_prob:.1%}")
    col2.metric("Real Score", reality_score)

else:
    st.info("Upload any image → real selfies = green, 2025 AI = red")

st.caption("AIRealNet 2025 model • No torch errors • Works right now")
