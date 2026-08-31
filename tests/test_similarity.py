"""Tests for src/similarity/similarity.py using synthetic embeddings only —
no real images, no network access, no CLIP model load required.
"""
import numpy as np

from src.similarity.similarity import cosine_similarity, cluster_dbscan, cluster_images, repetition_score


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _make_synthetic_embeddings(dim: int = 768, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """3 near-duplicate embeddings (small noise around one base vector), plus
    2 embeddings that are unrelated (orthogonal-ish random directions).
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim)

    near_dupes = np.stack([base + rng.normal(scale=0.01, size=dim) for _ in range(3)])
    unrelated = rng.normal(size=(2, dim))

    embeddings = _l2_normalize(np.concatenate([near_dupes, unrelated], axis=0))
    labels = ["dupe_0", "dupe_1", "dupe_2", "unrelated_0", "unrelated_1"]
    return embeddings, labels


def test_cosine_similarity_matrix_shape_and_self_similarity():
    embeddings, _ = _make_synthetic_embeddings()
    sim = cosine_similarity(embeddings)

    assert sim.shape == (5, 5)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-5)


def test_near_duplicates_cluster_together_with_dbscan():
    embeddings, _ = _make_synthetic_embeddings()

    cluster_ids = cluster_dbscan(embeddings, eps=0.05, min_samples=2)

    dupe_clusters = cluster_ids[:3]
    unrelated_clusters = cluster_ids[3:]

    assert len(set(dupe_clusters)) == 1
    assert unrelated_clusters[0] != dupe_clusters[0]
    assert unrelated_clusters[1] != dupe_clusters[0]
    assert unrelated_clusters[0] != unrelated_clusters[1]


def test_cluster_images_uses_config_dbscan_settings():
    embeddings, _ = _make_synthetic_embeddings()
    config = {
        "similarity": {
            "clustering_method": "dbscan",
            "dbscan_eps": 0.05,
            "dbscan_min_samples": 2,
        }
    }

    cluster_ids = cluster_images(embeddings, config)

    assert len(set(cluster_ids[:3])) == 1
    assert cluster_ids[3] != cluster_ids[0]
    assert cluster_ids[4] != cluster_ids[0]


def test_repetition_score_reflects_cluster_size():
    # 5 images total: 3 in one cluster, 2 singletons.
    cluster_ids = np.array([0, 0, 0, 1, 2])

    scores = repetition_score(cluster_ids)

    assert np.isclose(scores[0], 2 / 5)
    assert np.isclose(scores[1], 2 / 5)
    assert np.isclose(scores[2], 2 / 5)
    assert np.isclose(scores[3], 0.0)
    assert np.isclose(scores[4], 0.0)
