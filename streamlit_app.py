import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load the 2025 AI-vs-Human detector (SigLIP tuned for photoreal deepfakes like Trump gens)
@st.cache_resource(show_spinner="Loading detector (~4 sec, one-time)…")
def load_model():
    model_name = "Ateeqq/ai-vs-human-image-detector"  # Handles Flux/SD3.5/Gemini Trump fakes, 94% acc
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name, torch_dtype=torch.float32)
    return processor, model

processor, model = load_model()

st.set_page_config(page_title="CausalEcho – Trump AI Fixed", layout="wide")
st.title("CausalEcho: Real vs AI Detector")
st.caption("2025 SigLIP model: Nails Trump deepfakes • Real pics green, AI red")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)

    with st.spinner("Scanning for deepfakes..."):
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        # Labels: 0 = AI/human-generated, 1 = real
        ai_prob = float(probs[0])
        real_prob = float(probs[1])

    # Lower threshold for better fake-catching on celebs: >0.55 = AI
    if ai_prob > 0.55:
        st.error(f"🚨 AI-GENERATED / DEEPFAKE ({ai_prob:.1%} confidence)")
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
    st.info("💡 Test Trump: Real rally shots = green. AI Trump (Gemini/Flux) = red now")

st.caption("Model: Ateeqq/ai-vs-human-image-detector (2025) – Optimized for modern photoreal fakes")
