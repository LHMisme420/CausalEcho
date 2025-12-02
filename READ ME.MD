# CausalEcho — Physics Doesn't Lie

> "Deepfakes break pixels.  
> We break deepfakes with **physics**."

The first detector that doesn't hunt artifacts.  
It asks one question: **Could this have physically happened?**

No → **Acausal** → **Fake**.

## Features
- Physics-based causality checks
- Zero training data needed
- Live webcam mode
- Web app demo
- Works forever — no model retraining

## Run It Now
```bash
git clone https://github.com/LHMisme420/CausalEcho.git
cd CausalEcho
pip install -r requirements.txt

python causal_echo_detector.py "your_video.mp4"
# or
streamlit run streamlit_app.py   # ← easiest
