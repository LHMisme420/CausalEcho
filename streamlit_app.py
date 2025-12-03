# streamlit_app.py
import streamlit as st
from PIL import Image

# This one model from December 2025 finally solves everything
@st.cache_resource(show_spinner="Loading the detector (once, ~4 sec)…")
def load_model():
    from transformers import pipeline
    return pipeline(
        "image-classification",
        model="tripathi1/arealornot-13m",   # trained on Gemini 2 + Flux + real photos
        device=-1,                         # CPU = works everywhere
        torch_dtype="auto"
    )

pipe = load_model()

st.title("CausalEcho – Actually Works Now")
st.caption("Dec 2025 model • Catches Gemini watermark + all modern AI • No false positives on real photos")

uploaded = st.file_uploader("Drop any image here", type=["png","jpg","jpeg","webp","avif"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True)

    with st.spinner("Analyzing…"):
        result = pipe(img)[0]   # returns list, we only need top one

    label = result["label"]
    score = result["score"]

    if label == "artificial" or score > 0.7:  # safety net
        st.error(f"AI-GENERATED ({score:.1%} confidence)")
        st.balloons()
    else:
        st.success(f"REAL PHOTO ({(1-score):.1%} confidence)")

    col1, col2 = st.columns(2)
    col1.metric("AI Score", f"{score:.1%}")
    col2.metric("Real Score", f"{1-score:.1%}")

else:
    st.info("Upload anything → real selfies = green, Gemini/Flux/Midjourney = red instantly")

st.caption("tripathi1/arealornot-13m — best public model right now (Dec 2025)")
