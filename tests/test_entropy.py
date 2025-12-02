from causal_echo_detector import analyze_entropy

def test_real_entropy_near_natural(real_samples):
    for sample in real_samples:
        score = analyze_entropy(str(sample))
        assert score > 0.6

def test_synthetic_entropy_artifacts(synthetic_samples):
    for sample in synthetic_samples:
        score = analyze_entropy(str(sample))
        assert score < 0.7
