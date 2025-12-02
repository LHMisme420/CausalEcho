from causal_echo_detector import analyze_biological

def test_real_biological_sync(real_samples):
    for sample in real_samples:
        score = analyze_biological(str(sample))
        assert score > 0.65

def test_synthetic_biological_desync(synthetic_samples):
    for sample in synthetic_samples:
        score = analyze_biological(str(sample))
        assert score < 0.7
