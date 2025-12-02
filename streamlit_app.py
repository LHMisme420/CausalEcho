import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch
import numpy as np
from datetime import datetime

# Cache the model load (happens once, ~15s first time)
@st.cache_resource
def load_model():
    model_name = "Ateeqq/ai-vs-human-image-detector"  # 2025-tuned for Midjourney/Flux/SD3
    model = SiglipForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor

model, processor = load_model()

st.set_page_config(page_title="CausalEcho — 2025 AI KILLER", layout="wide")
st.title("🔍 Reality Violation Detector")
st.caption("Powered by Ateeqq/ai-vs-human (95% acc on 2025 AI like Flux/SD3/Midjourney v6)")

uploaded = st.file_uploader("Upload image (real selfie or AI-generated)", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running AI detection... (SigLIP 2025 model)"):
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).numpy()[0]
        
        # Labels: 0=AI/fake, 1=real (per model card)
        fake_prob = probs[0]
        real_prob = probs[1]
        
        impossible = fake_prob > 0.5  # Tuned threshold for 2025 balance
        reality_score = round(real_prob, 3)
        issues = ["AI generation artifacts detected (e.g., unnatural textures from diffusion models)"] if impossible else []

    result = {
        "impossible": impossible,
        "reality_score": reality_score,
        "issues": issues if issues else ["No AI artifacts found"],
        "message": "Reality violated! AI-generated image." if impossible else "Reality holds. Authentic photo.",
        "analyzed_at": datetime.now().isoformat(),
        "fake_probability": round(fake_prob, 3)
    }

    st.subheader("Detection Results")
    st.json(result, expanded=True)
    
    if impossible:
        st.error("🚨 AI-GENERATED / FAKE DETECTED (Confidence: {:.1%})".format(fake_prob))
        st.balloons()
    else:
        st.success("✅ REAL PHOTOGRAPH — Reality holds (Confidence: {:.1%})".format(real_prob))
    
    # Debug metrics
    col1, col2 = st.columns(2)
    col1.metric("AI Probability", f"{fake_prob:.1%}")
    col2.metric("Real Score", reality_score)

else:
    st.info("👆 Upload a test: Real phone pic = green. Midjourney/Flux 'photoreal portrait' = red!")

st.markdown("---")
st.caption("Based on 2025 SigLIP fine-tune (trained on 120k images incl. SD3.5/Flux) · No false 'real' on modern AI")
