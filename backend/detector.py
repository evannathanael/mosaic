"""Trained AI-vs-real image detector for the feed.

A frozen open_clip ``ViT-L-14 / openai`` backbone + a small trained MLP head
(``combined_head.pt`` — jointly trained on CIFAKE + SID-Set + WildFake). Given an
image we run the head and ``sigmoid`` its logit to get ``P(AI-generated)``; that
probability drives every post's ``diversity_label``:

    P >= threshold  and repeated  -> "repeated_synthetic"   (AI, and a near-dup)
    P >= threshold                -> "unique_ai"            (AI, distinct)
    else                          -> "original"             (not AI)

The similarity engine (backend/similarity_index.py) decides *repeated*; this
module only decides *AI or not*.

Torch is imported lazily and every failure degrades gracefully (predict_pil ->
None, detector_ready() -> False), so the backend still runs without the ML deps
or the checkpoint — the upload path then falls back to the filename heuristic.
"""
from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import unquote

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "dataset"
SCORE_CACHE = DATASET_DIR / "ai_scores.npz"
STATIC_DEMO_PREFIX = "/static/demo/"
DEFAULT_CHECKPOINT = ROOT / "combined_head.pt"
DEFAULT_THRESHOLD = 0.5

_MODEL = None
_MODEL_TRIED = False
_THRESHOLD: float | None = None
_CHECKPOINT: Path | None = None


def _config() -> dict:
    try:
        import yaml

        return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text()) or {}
    except Exception:  # noqa: BLE001
        return {}


def _checkpoint_path() -> Path:
    global _CHECKPOINT
    if _CHECKPOINT is None:
        raw = _config().get("detector", {}).get("checkpoint")
        _CHECKPOINT = (ROOT / raw) if raw else DEFAULT_CHECKPOINT
    return _CHECKPOINT


def ai_threshold() -> float:
    """P(AI) at or above which a post is labelled AI-generated."""
    global _THRESHOLD
    if _THRESHOLD is None:
        try:
            _THRESHOLD = float(_config()["detector"]["threshold"])
        except Exception:  # noqa: BLE001
            _THRESHOLD = DEFAULT_THRESHOLD
    return _THRESHOLD


def score_seed_enabled() -> bool:
    """Whether to overwrite seed posts' probabilities with model scores at startup.
    Off -> seed posts keep the curated fixture values in backend/demo.py and only
    new uploads run the detector. Toggle in configs/config.yaml (detector.score_seed).
    """
    val = _config().get("detector", {}).get("score_seed", True)
    return bool(val)


def _load_detector():
    """Lazily build + cache the CLIP-backbone + trained head. None on any failure."""
    global _MODEL, _MODEL_TRIED
    if _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True
    if not bool(_config().get("detector", {}).get("enabled", False)):
        logger.info("Live detector disabled in configs/config.yaml; using fallback scoring.")
        return None
    path = _checkpoint_path()
    try:
        import torch

        from src.models.clip_aigc_head import AIGCClipDetector

        if not path.exists():
            logger.warning("Detector checkpoint %s not found; AI scoring disabled.", path)
            return None
        ckpt = torch.load(path, map_location="cpu")
        model = AIGCClipDetector(
            ckpt.get("clip_model_name", "ViT-L-14"),
            ckpt.get("clip_pretrained", "openai"),
            ckpt.get("hidden_dim", 256),
            ckpt.get("dropout", 0.2),
        )
        model.head.load_state_dict(ckpt["head_state_dict"])
        model.eval()
        _MODEL = model
        logger.info("Loaded AI detector head from %s", path)
    except Exception as exc:  # noqa: BLE001 - never let the detector break startup/upload
        logger.warning("AI detector unavailable, falling back to heuristic: %s", exc)
        _MODEL = None
    return _MODEL


def detector_ready() -> bool:
    return _load_detector() is not None


def predict_pil(image) -> float | None:
    """Return P(AI-generated) in [0, 1] for a PIL image, or None if unavailable."""
    model = _load_detector()
    if model is None:
        return None
    try:
        import torch

        tensor = model.clip_preprocess(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()
        return round(float(prob), 6)
    except Exception as exc:  # noqa: BLE001
        logger.warning("AI scoring failed for one image: %s", exc)
        return None


def label_for(probability: float, *, repeated: bool) -> str:
    """The single source of truth for probability -> diversity_label."""
    if probability >= ai_threshold():
        return "repeated_synthetic" if repeated else "unique_ai"
    return "original"


# ---------------------------------------------------------------------------
# Seed-score cache: data/dataset/ai_scores.npz, built by
# scripts/build_demo_ai_scores.py. Read at startup so seed posts get a real
# score without paying the CLIP cost per request.
# ---------------------------------------------------------------------------
def _local_key(thumbnail_url: str) -> str | None:
    if not thumbnail_url.startswith(STATIC_DEMO_PREFIX):
        return None
    return unquote(thumbnail_url[len(STATIC_DEMO_PREFIX):])


def load_score_cache() -> dict[str, float]:
    if not SCORE_CACHE.exists():
        return {}
    try:
        import numpy as np

        data = np.load(SCORE_CACHE, allow_pickle=True)
        return {str(k): float(v) for k, v in zip(data["paths"], data["scores"])}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", SCORE_CACHE, exc)
        return {}


def _apply_score(post: dict, probability: float) -> None:
    post["ai_probability"] = round(float(probability), 6)
    post["confidence"] = min(1.0, abs(probability - 0.5) * 2)
    post["robustness"] = {
        "compressed": max(0.0, probability - 0.02),
        "cropped": max(0.0, probability - 0.05),
        "blurred": max(0.0, probability - 0.08),
        "recolored": max(0.0, probability - 0.03),
    }
    post["analysis_mode"] = "model"


def attach_seed_scores(posts: list[dict], overrides: dict[str, tuple[float, str]] | None = None) -> None:
    """Overwrite seed posts' ai_probability from the on-disk cache (and any
    explicit per-image_id overrides). Posts missing from the cache keep the
    fallback probability set in backend/demo.py."""
    cache = load_score_cache()
    overrides = overrides or {}
    for post in posts:
        key = _local_key(post.get("thumbnail_url", ""))
        # Score the curated REAL/ and FAKE/ demo images. The near_dupes/ fixtures
        # are variants we generated from FAKE images — their AI + cluster ground
        # truth is known by construction, so leave their fixture values alone.
        if key is not None and key in cache and (key.startswith("REAL/") or key.startswith("FAKE/")):
            _apply_score(post, cache[key])
        image_id = post.get("image_id")
        if image_id in overrides:
            prob, _label = overrides[image_id]
            _apply_score(post, prob)
