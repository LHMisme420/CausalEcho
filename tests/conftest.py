import pytest
from pathlib import Path

@pytest.fixture
def real_samples():
    return list(Path("tests/samples/real").glob("*"))

@pytest.fixture
def synthetic_samples():
    return list(Path("tests/samples/synthetic").glob("*"))
