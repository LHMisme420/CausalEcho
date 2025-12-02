from causal_echo_detector import analyze_physical

def test_real_media_physical_coherence(real_samples):
    for sample in real_samples:
        score = analyze_physical(str(sample))
        assert score > 0.65

def test_synthetic_media_physical_failure(synthetic_samples):
    for sample in synthetic_samples:
        score = analyze_physical(str(sample))
        assert score < 0.7
