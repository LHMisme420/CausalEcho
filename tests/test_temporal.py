from causal_echo_detector import analyze_temporal

def test_real_media_temporal_integrity(real_samples):
    for sample in real_samples:
        score = analyze_temporal(str(sample))
        assert score > 0.65, f"Temporal failed on real sample: {sample}"

def test_synthetic_media_temporal_failure(synthetic_samples):
    for sample in synthetic_samples:
        score = analyze_temporal(str(sample))
        assert score < 0.7, f"Temporal falsely passed on fake: {sample}"
