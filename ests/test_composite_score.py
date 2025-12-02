from causal_echo_detector import verify_media

def test_real_media_scores_high(real_samples):
    for sample in real_samples:
        result = verify_media(str(sample))
        assert result["causality_score"] > 0.7
        assert result["verdict"] in ["Highly Authentic", "Likely Authentic"]

def test_synthetic_media_scores_low(synthetic_samples):
    for sample in synthetic_samples:
        result = verify_media(str(sample))
        assert result["causality_score"] < 0.6
        assert result["verdict"] in ["Likely Synthetic", "Highly Synthetic"]
