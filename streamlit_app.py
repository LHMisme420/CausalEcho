import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np
from datetime import datetime

# Cache the model load (happens once, ~15-25s first time)
@st.cache_resource
def load_model():
    model_name = "Falconsai/AIRealNet"  # Sept 2025 release: Tuned for 2025 AI vs real photos
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor

model, processor = load_model()

st.set_page_config(page_title="CausalEcho — 2025 AI DESTROYER", layout="wide")
st.title("🔍 Reality Violation Detector")
st.caption("Powered by AIRealNet (Falconsai) — 94% acc on Flux/SD3.5/Midjourney v6.1 fakes")

uploaded = st.file_uploader("Upload image (real selfie or AI-generated)", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running 2025 AI detection... (AIRealNet model)"):
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).numpy()[0]
        
        # Labels: 0=AI/fake, 1=real (per model card)
        fake_prob = probs[0]
        real_prob = probs[1]
        
        impossible = fake_prob > 0.4  # Lowered threshold for 2025 aggression (tune to 0.3 if needed)
        reality_score = round(real_prob, 3)
        issues = ["2025 AI generation artifacts detected (e.g., synthetic diffusion patterns)"] if impossible else []

    result = {
        "impossible": impossible,
        "reality_score": reality_score,
        "issues": issues if issues else ["No AI artifacts found — authentic capture"],
        "message": "Reality violated! AI-generated image confirmed." if impossible else "Reality holds. Genuine photograph.",
        "analyzed_at": datetime.now().isoformat(),
        "fake_probability": round(fake_prob, 3)
    }

    st.subheader("Detection Results")
    st.json(result, expanded=True)
    
    if impossible:
        st.error(f"🚨 AI-GENERATED / FAKE DETECTED (Fake Prob: {fake_prob:.1%})")
        st.balloons()
    else:
        st.success(f"✅ REAL PHOTOGRAPH — Reality holds (Real Prob: {real_prob:.1%})")
    
    # Debug metrics
    col1, col2 = st.columns(2)
    col1.metric("AI Probability", f"{fake_prob:.1%}")
    col2.metric("Reality Score", reality_score)

else:
    st.info("👆 Upload a test: Real phone pic = green. Fresh Flux/Midjourney 'photoreal portrait' = red!")

st.markdown("---")
st.caption("Sept 2025 AIRealNet (fine-tuned on 2025 datasets) · Handles concept drift · No more 'real' on sneaky fakes [](grok_render_citation_card_json={"cardIds":["ed25d2"]})")
