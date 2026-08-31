"""CLIP-embedding similarity index for the feed.

Every post carries a 768-d L2-normalized embedding. On upload we embed the new
image with the shared CLIP backbone (src/similarity/embed.py) and assign its
``similarity_cluster`` by cosine similarity to existing posts — so a re-post or a
lightly-edited near-duplicate lands in the same cluster as the original, and the
feed ranker keeps them apart.

Torch is imported lazily and every failure degrades gracefully (embedding =
None), so the backend still runs in an environment without the ML deps.
"""
from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import unquote

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "dataset"
EMBEDDING_CACHE = DATASET_DIR / "embeddings.npz"
STATIC_DEMO_PREFIX = "/static/demo/"

_THRESHOLD: float | None = None


def _ml_enabled() -> bool:
    """Whether live CLIP embedding is enabled for uploads.

    Seed embeddings can still be loaded from the committed cache when this is
    disabled. Keeping live inference opt-in prevents a memory-constrained demo
    machine from attempting to allocate ViT-L/14 on every first upload.
    """
    try:
        import yaml

        cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text()) or {}
        return bool(cfg.get("similarity", {}).get("enabled", False))
    except Exception:
        return False


def _threshold() -> float:
    global _THRESHOLD
    if _THRESHOLD is None:
        try:
            import yaml

            cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
            _THRESHOLD = float(cfg["similarity"]["threshold"])
        except Exception:
            _THRESHOLD = 0.90
    return _THRESHOLD


def _local_path(thumbnail_url: str) -> Path | None:
    if not thumbnail_url.startswith(STATIC_DEMO_PREFIX):
        return None
    return DATASET_DIR / unquote(thumbnail_url[len(STATIC_DEMO_PREFIX):])


def embed_image(image) -> list[float] | None:
    """Embed a PIL image via the shared CLIP backbone. None if ML deps/model unavailable."""
    if not _ml_enabled():
        return None
    try:
        from src.similarity.embed import embed_pil

        return [round(float(x), 6) for x in embed_pil(image)]
    except Exception as exc:  # noqa: BLE001 - never let embedding break an upload
        logger.warning("Embedding unavailable, falling back to hash-only clustering: %s", exc)
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    # both vectors are already L2-normalized
    return sum(x * y for x, y in zip(a, b))


def assign_cluster(embedding: list[float] | None, posts: list[dict], next_cluster_fn) -> tuple[int, float]:
    """Return (cluster_id, best_similarity). Falls back to next_cluster_fn() when
    there is no embedding or nothing similar enough."""
    if embedding is None:
        return next_cluster_fn(), 0.0
    best_sim = 0.0
    best_cluster: int | None = None
    for post in posts:
        other = post.get("embedding")
        if not other:
            continue
        sim = _cosine(embedding, other)
        if sim > best_sim:
            best_sim = sim
            best_cluster = post["similarity_cluster"]
    if best_cluster is not None and best_sim >= _threshold():
        return best_cluster, best_sim
    return next_cluster_fn(), best_sim


def annotate_cluster_similarity(posts: list[dict]) -> None:
    """Set ``post['similarity_score']`` = the max cosine similarity of this image
    to any other post in the same cluster (0.0 if it's alone or has no embedding).
    This is the real image-to-image number the UI shows, distinct from the
    synthetic ``repetition_score``."""
    from collections import defaultdict

    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for post in posts:
        by_cluster[post["similarity_cluster"]].append(post)

    for members in by_cluster.values():
        for post in members:
            emb = post.get("embedding")
            if not emb:
                post.setdefault("similarity_score", 0.0)
                continue
            best = 0.0
            for other in members:
                if other is post:
                    continue
                other_emb = other.get("embedding")
                if other_emb:
                    best = max(best, _cosine(emb, other_emb))
            post["similarity_score"] = round(best, 4)


def load_embedding_cache() -> dict[str, list[float]]:
    """Read the committed seed-embedding cache (data/dataset/embeddings.npz).

    Empty dict if the cache is absent — run scripts/build_demo_embeddings.py to
    build it. Never computes here, so server startup and tests stay fast.
    """
    if not EMBEDDING_CACHE.exists():
        return {}
    try:
        import numpy as np

        data = np.load(EMBEDDING_CACHE, allow_pickle=True)
        return {str(k): [float(x) for x in v] for k, v in zip(data["paths"], data["embeddings"])}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", EMBEDDING_CACHE, exc)
        return {}


def attach_seed_embeddings(posts: list[dict]) -> None:
    """Populate ``post['embedding']`` for seed posts from the on-disk cache."""
    cache = load_embedding_cache()
    if not cache:
        return
    for post in posts:
        path = _local_path(post.get("thumbnail_url", ""))
        if not path:
            continue
        key = path.relative_to(DATASET_DIR).as_posix()
        if key in cache:
            post["embedding"] = cache[key]
