"""Quick smoke tests — not full unit tests, just enough to catch broken
imports or obvious pipeline breakage early. Run with: pytest tests/
"""
import numpy as np
import pytest


def test_config_loads():
    from src.utils import load_config

    config = load_config("configs/config.yaml")
    assert "model" in config
    assert "training" in config
    assert config["run_name"] == "baseline"


def test_experiment_override():
    from src.utils import load_config

    config = load_config("configs/config.yaml", experiment="rotation_jitter")
    assert config["run_name"] == "rotation_jitter"
    assert config["train_only_augmentation"]["random_rotation_degrees"] == 15


def test_transforms_run_without_error():
    from src.data.transforms import NAMED_EVAL_TRANSFORMS, named_eval_transform

    fake_img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    for name in NAMED_EVAL_TRANSFORMS:
        out = named_eval_transform(name, fake_img)
        assert out.dtype == np.uint8
        assert out.ndim == 3


def test_classifier_head_shapes():
    import torch
    from src.models.classifier import ClassifierHead

    head = ClassifierHead(embedding_dim=512, hidden_dim=128)
    fake_embedding = torch.randn(4, 512)
    out = head(fake_embedding)
    assert out.shape == (4,)


def test_union_find_clustering_logic():
    from src.similarity.clustering import cluster_threshold
    import numpy as np

    # 3 near-identical embeddings + 1 distinct one -> expect 2 clusters
    base = np.random.rand(512)
    embeddings = np.stack([
        base + np.random.normal(0, 0.001, 512),
        base + np.random.normal(0, 0.001, 512),
        base + np.random.normal(0, 0.001, 512),
        np.random.rand(512),
    ])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    clusters = cluster_threshold(embeddings, threshold=0.99)
    assert len(set(clusters[:3])) == 1  # first three should cluster together
