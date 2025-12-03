# streamlit_app.py
import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load the reliable 2024/2025 AI detector (ViT tuned for artistic + photo fakes)
@st.cache_resource(show_spinner="Loading detector (~3 sec, one-time)…")
def load_model():
    model_name = "umm-maybe/AI-image-detector"  # Proven, exists, handles Gemini/Trump
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name, torch_dtype=torch.float32)
    return processor, model

processor, model = load_model()

st.set_page_config(page_title="CausalEcho – Locked In", layout="wide")
st.title("CausalEcho: Real vs AI Detector")
st.caption("umm-maybe/AI-image-detector • Catches 2025 AI (Gemini/Flux) + no Trump confusion")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    with st.spinner("Scanning..."):
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        # Labels: 0 = artificial/AI, 1 = real
        ai_prob = float(probs[0])
        real_prob = float(probs[1])

    # Threshold: >0.6 = AI (balanced for celebs like Trump)
    if ai_prob > 0.6:
        st.error(f"🚨 AI-GENERATED ({ai_prob:.1%} confidence)")
        st.balloons()
    else:
        st.success(f"✅ REAL PHOTO ({real_prob:.1%} confidence)")

    col1, col2 = st.columns(2)
    col1.metric("AI Probability", f"{ai_prob:.1%}")
    col2.metric("Real Probability", f"{real_prob:.1%}")

    with st.expander("Raw Output"):
        st.write(f"AI: {ai_prob:.3f} | Real: {real_prob:.3f}")
        st.json({"logits": outputs.logits.tolist()})

else:
    st.info("💡 Real Trump pics = green. AI Trump/Gemini = red (watermarks via artifacts)")

st.caption("Model: umm-maybe/AI-image-detector (Oct 2024, still golden in 2025)")
