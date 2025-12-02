/core/
  temporal.py
  physics.py
  biology.py
  entropy.py
engine.py
api.py
from core.temporal import check_temporal
from core.physics import check_physics
from core.biology import check_biology

def verify(path):
    t = check_temporal(path)
    p = check_physics(path)
    b = check_biology(path)

    score = (t + p + b) / 3
    return {
        "score": score,
        "verdict": "Likely Authentic" if score > 0.7 else "Likely Synthetic"
    }
