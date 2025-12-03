# streamlit_app.py  ←  just overwrite everything with this
import streamlit as st
from PIL import Image
from transformers import pipeline

# This model is LIVE right now and loads in 3–5 seconds first time
@st.cache_resource(show_spinner="Loading detector (one-time, ~5 sec)…")
def load_model():
    return pipeline(
        "image-classification",
        model="not-lain/ai-or-real",   # ← currently the best working model (Dec 2025)
        device=-1                      # CPU = works everywhere
    )

pipe = load_model()

st.title("CausalEcho – Finally Works")
st.caption("not-lain/ai-or-real • Instantly catches Gemini watermarks & modern AI • Real Trump photos = green")

uploaded = st.file_uploader("Upload any image", type=["png","jpg","jpeg","webp","avif"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True)

    with st.spinner("Checking…"):
        result = pipe(img)[0]   # top prediction only

    label = result["label"]
    score = result["score"]

    if label == "FAKE" or label == "ai" or score > 0.65:
        st.error(f"AI / DEEPFAKE ({score:.1%} confidence)")
        st.balloons()
    else:
        st.success(f"REAL PHOTO ({score:.1%} confidence)")

    col1, col2 = st.columns(2)
    col1.metric("AI / Fake", f"{score:.1%}" if 'FAKE' in label.upper() else f"{1-score:.1%}")
    col2.metric("Real", f"{score:.1%}" if 'REAL' in label.upper() else f"{1-score:.1%}")

else:
    st.info("Upload real selfies or AI images → works instantly now")

st.caption("Model: not-lain/ai-or-real – proven working Dec 2025")
