# CausalEcho

**Deepfakes don’t just lie — they break the universe.**  
CausalEcho doesn’t *detect* deepfakes. It **disproves** them by catching violations of physics, causality, and reality itself.

![CausalEcho in action](https://github.com/user-attachments/assets/placeholder-gif-link-when-you-have-one)
*↑ A deepfake claiming water flows upward. CausalEcho flags impossible gravity in < 0.5 s.*

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

## Why this approach wins

Every other detector plays whack-a-mole with neural artifacts.  
We ask one question no AI can ever fake perfectly:

> **"Does this scene obey the unbreakable laws of physics and causality?"**

Light coming from nowhere? Shadows moving backward? Effect before cause?  
→ Physically impossible → **disproven**.

## Current Causality Checks (v0.1)

| Check                          | Detects                                           | Status     |
|-------------------------------|----------------------------------------------------|------------|
| Gravity & free-fall            | Objects falling up / wrong acceleration            | ✅ Done    |
| Impossible light transport     | Negative light, light from nowhere                 | ✅ Done    |
| Shadow inconsistencies         | Detached or impossible shadows                     | ✅ Done    |
| Reverse causation              | Effect before cause (e.g., glass breaks → bullet)  | ✅ Done    |
| Audio-visual causality breach  | Sound arrives before visible event                 | ⚡ In progress |
| Conservation of momentum       | Teleportation, infinite acceleration              | 🔜 Planned |

## Live Demo

🚀 **Try it now:** https://causalecho.streamlit.app  
*(Free Streamlit Community Cloud — deploy in 30 seconds, instructions below)*

## Quick Start (Local)

```bash
git clone https://github.com/LHMisme420/CausalEcho.git
cd CausalEcho
pip install -r requirements.txt
streamlit run streamlit_app.py
