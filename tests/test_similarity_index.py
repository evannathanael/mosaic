"""Cluster assignment by cosine similarity (no model needed — synthetic vectors)."""

from backend.similarity_index import _cosine, assign_cluster


def _unit(*values: float) -> list[float]:
    import math

    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def test_cosine_of_identical_vectors_is_one():
    v = _unit(1.0, 2.0, 3.0)
    assert abs(_cosine(v, v) - 1.0) < 1e-9


def test_assign_cluster_reuses_similar_post():
    a = _unit(1.0, 0.0, 0.0)
    near = _unit(0.99, 0.14, 0.0)  # cosine ~0.99 with a
    posts = [{"similarity_cluster": 42, "embedding": a}]
    cluster, sim = assign_cluster(near, posts, next_cluster_fn=lambda: 999)
    assert cluster == 42
    assert sim > 0.9


def test_assign_cluster_falls_back_when_dissimilar():
    a = _unit(1.0, 0.0, 0.0)
    orthogonal = _unit(0.0, 1.0, 0.0)
    posts = [{"similarity_cluster": 42, "embedding": a}]
    cluster, _ = assign_cluster(orthogonal, posts, next_cluster_fn=lambda: 999)
    assert cluster == 999


def test_assign_cluster_without_embedding_uses_fallback():
    cluster, sim = assign_cluster(None, [], next_cluster_fn=lambda: 7)
    assert cluster == 7 and sim == 0.0
