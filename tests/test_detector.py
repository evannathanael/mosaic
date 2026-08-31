"""Trained AI-vs-real detector wiring (backend/detector.py)."""

import io

import pytest
from PIL import Image

from backend import detector


def _png() -> Image.Image:
    return Image.new("RGB", (32, 32), (120, 30, 200))


def test_label_for_mapping():
    t = detector.ai_threshold()
    assert detector.label_for(t + 0.01, repeated=False) == "unique_ai"
    assert detector.label_for(t + 0.01, repeated=True) == "repeated_synthetic"
    assert detector.label_for(t - 0.01, repeated=True) == "original"
    assert detector.label_for(t - 0.01, repeated=False) == "original"


def test_load_score_cache_shape():
    cache = detector.load_score_cache()
    assert isinstance(cache, dict)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in cache.values())


@pytest.mark.skipif(not detector.detector_ready(), reason="combined_head.pt / torch not available")
def test_predict_pil_returns_probability():
    prob = detector.predict_pil(_png())
    assert prob is not None
    assert 0.0 <= prob <= 1.0
