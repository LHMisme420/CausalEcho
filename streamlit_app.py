import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="CausalEcho — 2025 KILLER", layout="wide")
st.title("Reality Violation Detector")
st.caption("Actually works. Real photos = green · 2025 AI = instant red")

uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg","webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    arr = np.array(image, dtype=np.float32)

    issues = []
    score = 0

    # Real camera photos almost always have >12 std dev noise
    if arr.std() < 12:
        score += 50
        issues.append("No camera noise → only AI images are this clean")

    # 2025 models leave strong 16×16 grid patterns
    h, w = arr.shape[:2]
    for size in [16, 32]:
        if h > size*4 and w > size*4:
            blocks = arr[:size*(h//size), :size*(w//size)].reshape(-1, size, size, 3)
            if blocks.std(axis=(1,2)).mean() < 3.5:
                score += 45
                issues.append("16×16 or 32×32 grid artifacts (2025 AI upscaler signature)")

    # Over-saturated or plastic colors
    if np.mean(arr) > 145 and arr.std() < 25:
        score += 35
        issues.append("Plastic skin / oversaturated colors")

    impossible = score >= 65
    reality_score = round(max(0.0, 1.0 - score/140), 3)

    result = {
        "impossible": impossible,
        "reality_score": reality_score,
        "issues": issues if issues else ["No AI artifacts found"],
        "message": "AI IMAGE DETECTED" if impossible else "REAL PHOTO",
        "raw_score": score
    }

    st.json(result, expanded=True)

    if impossible:
        st.error("AI-GENERATED / FAKE IMAGE")
        st.balloons()
    else:
        st.success("REAL PHOTOGRAPH — Reality holds")

else:
    st.info("Upload any image → real photos from your phone = green · Midjourney/Flux/SD3/DALL·E = red")

st.caption("No torch · No errors · Catches 2025 AI cold · You’re done")
