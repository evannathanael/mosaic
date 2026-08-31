"""The near-duplicate demo fixtures should seed one cluster per source image."""

from collections import Counter

import pytest

from backend.demo import NEAR_DUPES_MANIFEST, NEAR_DUPE_CLUSTER_OFFSET, load_seed_posts


requires_fixtures = pytest.mark.skipif(
    not NEAR_DUPES_MANIFEST.exists(),
    reason="run scripts/build_demo_dupes.py to generate data/dataset/near_dupes/",
)


@requires_fixtures
def test_near_dupe_posts_group_by_cluster():
    posts = load_seed_posts()
    near_dupes = [p for p in posts if p["source"] == "demo_dataset" and p["handle"].startswith("@reposter")]
    assert near_dupes, "expected seeded near-duplicate posts"

    sizes = Counter(p["similarity_cluster"] for p in near_dupes)
    assert all(cluster >= NEAR_DUPE_CLUSTER_OFFSET for cluster in sizes)
    assert all(size >= 2 for size in sizes.values())

    for post in near_dupes:
        assert post["diversity_label"] == "repeated_synthetic"
        assert 0.0 < post["repetition_score"] < 1.0
        # one handle + account per cluster
        assert post["handle"] == f"@reposter{post['similarity_cluster'] - NEAR_DUPE_CLUSTER_OFFSET:04d}"


@requires_fixtures
def test_every_seed_post_has_integer_cluster():
    for post in load_seed_posts():
        assert isinstance(post["similarity_cluster"], int)
        assert 0.0 <= post["ai_probability"] <= 1.0
        assert post["diversity_label"] in {"original", "unique_ai", "repeated_synthetic"}
