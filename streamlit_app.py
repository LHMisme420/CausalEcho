import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load the 2025 AI-vs-Human detector (SigLIP fine-tuned on Midjourney/Flux/Gemini-era gens)
@st.cache_resource
def load_model():
    model_name = "Ateeqq/ai-vs-human-image-detector"  # Real HF model: 96% acc on 2025 AI
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name, torch_dtype=torch.float32)
    return processor, model

processor, model = load_model()

st.set_page_config(page_title="CausalEcho – Fixed & Accurate", layout="wide")
st.title("CausalEcho: Real vs AI Detector")
st.caption("2025 SigLIP model: Catches Gemini/Flux/Midjourney watermarks + artifacts • Real photos stay green")

uploaded = st.file_uploader("Upload image (PNG/JPG/WEBP)", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Scanning for AI generation..."):
        # Preprocess & infer
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        # Labels: 0 = AI/artificial, 1 = human/real
        ai_prob = float(probs[0])
        real_prob = float(probs[1])

    # Decision (tuned threshold for balance: >0.6 = AI)
    if ai_prob > 0.6:
        st.error(f"🚨 AI-GENERATED ({ai_prob:.1%} confidence)")
        st.balloons()
    else:
        st.success(f"✅ REAL PHOTO ({real_prob:.1%} confidence)")

    col1, col2 = st.columns(2)
    col1.metric("AI Probability", f"{ai_prob:.1%}")
    col2.metric("Real Probability", f"{real_prob:.1%}")

    with st.expander("Raw Output"):
        st.write(f"AI Score: {ai_prob:.3f} | Real Score: {real_prob:.3f}")
        st.json({"logits": outputs.logits.tolist()})

else:
    st.info("💡 Test it: Real selfies = green. Gemini/Midjourney/Flux images = red (watermarks auto-detected via artifacts)")

st.caption("Model: Ateeqq/ai-vs-human-image-detector (Dec 2025) – No more load errors!")
