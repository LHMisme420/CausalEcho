import streamlit as st
from PIL import Image

st.set_page_config(page_title="CausalEcho")
st.title("Reality Violation Detector")
st.write("THIS VERSION IS 100 % HARD-CODED TO PROVE THE APP IS ALIVE")

file = st.file_uploader("Upload anything", type=["png","jpg","jpeg","webp"])

if file:
    img = Image.open(file)
    st.image(img, use_column_width=True)
    
    st.json({
        "impossible": True,
        "reality_score": 0.000,
        "issues": ["← THIS IS A TEST ←", "If you see this = app is working", "Real detector coming in 10 seconds"],
        "message": "TEST SUCCESS – APP IS FIXED"
    })
    st.error("APP IS NOW UNDER CONTROL")
    st.balloons()
else:
    st.success("App is running perfectly. Upload any picture → you WILL see red alert above.")
