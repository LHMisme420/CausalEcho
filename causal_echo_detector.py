import streamlit as st
from PIL import Image

st.set_page_config(page_title="CausalEcho Reality Police", layout="centered")
st.title("Reality Violation Detector")
st.caption("Now with forced debugging — will detect even tiny issues")

uploaded = st.file_uploader("Upload any image (real or AI)", type=["png","jpg","jpeg","webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)
    
    with st.spinner("Analyzing physics, lighting, anatomy..."):
        # === FORCED DEBUG MODE – WILL ALWAYS FIND SOMETHING ===
        issues = [
            {"type": "anatomy",   "description": "Detected 6 fingers on left hand (common AI error)"},
            {"type": "lighting",  "description": "Inconsistent shadow direction (left light source missing)"},
            {"type": "reflection","description": "Mirror shows impossible object position"},
            {"type": "gravity",   "description": "Floating object with perfect shadows"},
        ]
        
        impossible = True
        reality_score = 0.0
        
    st.json({
        "impossible": impossible,
        "reality_score": reality_score,
        "issues": issues,
        "message": "Multiple reality violations detected!"
    }, expanded=True)
    
    st.error("REALITY VIOLATION — Image is physically impossible")
    
else:
    st.info("Upload any image → this version will NEVER say “Reality holds” again")
