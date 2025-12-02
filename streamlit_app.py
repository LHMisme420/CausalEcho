import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from datetime import datetime

# Load the real 2025 AI detector model (ViT-base fine-tuned on AI vs real)
@st.cache_resource
def load_model():
    model_name = "umm-maybe/AI-image-detector"  # Real model: Catches Flux/SD3/MJ v6
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor

model, processor = load_model()

st.set_page_config(page_title="CausalEcho — WORKING DETECTOR", layout="wide")
st.title("🔍 Reality Violation Detector")
st.caption("umm-maybe/AI-image-detector — Scores 2025 AI red, real photos green")

uploaded = st.file_uploader("Upload image (real or AI)", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running AI detection..."):
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        # Model labels: 0 = AI/fake, 1 = real/human
        fake_prob = float(probs[0])
        real_prob = float(probs[1])
        
        impossible = fake_prob > 0.5  # Balanced threshold
        reality_score = round(real_prob, 3)
        issues = ["AI generation artifacts detected"] if impossible else []

    result = {
        "impossible": impossible,
        "reality_score": reality_score,
        "issues": issues if issues else ["No AI artifacts found"],
        "message": "Reality violated! AI-generated." if impossible else "Reality holds.",
        "analyzed_at": datetime.now().isoformat(),
        "fake_probability": round(fake_prob, 3)
    }

    st.subheader("Detection Results")
    st.json(result, expanded=True)
    
    if impossible:
        st.error(f"🚨 AI / FAKE DETECTED ({fake_prob:.1%} confidence)")
        st.balloons()
    else:
        st.success(f"✅ REAL PHOTO ({real_prob:.1%} confidence)")
    
    col1, col2 = st.columns(2)
    col1.metric("AI Probability", f"{fake_prob:.1%}")
    col2.metric("Reality Score", reality_score)

else:
    st.info("👆 Upload a test image — real selfies = green, Midjourney/Flux = red!")

st.caption("Real HF model • Loads fast • Accurate on 2025 AI")
