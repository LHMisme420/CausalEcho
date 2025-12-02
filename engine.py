from core.temporal import analyze_temporal
from core.physical import analyze_physical
from core.biological import analyze_biological
from core.entropy import analyze_entropy


def verify_media(media_path: str) -> dict:
    T = analyze_temporal(media_path)
    P = analyze_physical(media_path)
    B = analyze_biological(media_path)
    E = analyze_entropy(media_path)

    causality_score = round((T + P + B + E) / 4, 4)

    if causality_score >= 0.85:
        verdict = "Highly Authentic"
    elif causality_score >= 0.70:
        verdict = "Likely Authentic"
    elif causality_score >= 0.50:
        verdict = "Indeterminate"
    elif causality_score >= 0.30:
        verdict = "Likely Synthetic"
    else:
        verdict = "Highly Synthetic"

    return {
        "media": media_path,
        "temporal_integrity": T,
        "physical_coherence": P,
        "biological_sync": B,
        "entropy_consistency": E,
        "causality_score": causality_score,
        "verdict": verdict
    }
