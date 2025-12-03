# inference.py
import json
import os
from datetime import datetime

# --- CONFIGURABLE THRESHOLDS (CRITICAL) ---
AI_THRESHOLD = 0.85       # Only accuse AI above this
REAL_THRESHOLD = 0.40     # Only assert REAL below this
UNCERTAIN_RANGE = (REAL_THRESHOLD, AI_THRESHOLD)

FORENSICS_LOG = "forensics_fp_log.jsonl"


def classify_with_uncertainty(ai_probability: float) -> str:
    """
    Converts raw model probability into:
    - AI
    - REAL
    - UNCERTAIN
    """

    if ai_probability >= AI_THRESHOLD:
        return "AI"
    elif ai_probability <= REAL_THRESHOLD:
        return "REAL"
    else:
        return "UNCERTAIN"


def log_forensic_event(
    image_id: str,
    ai_probability: float,
    predicted_label: str,
    ground_truth: str | None = None,
):
    """
    Logs false positives and uncertain cases for retraining & audit.
    """

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "image_id": image_id,
        "ai_probability": round(ai_probability, 4),
        "predicted_label": predicted_label,
        "ground_truth": ground_truth,
    }

    with open(FORENSICS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
