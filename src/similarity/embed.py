"""Extract embeddings from the SAME shared backbone used by the classifier —
no separate model, no extra training required. Cosine similarity between
these embeddings is what powers near-duplicate detection.

Preprocessing uses open_clip's own default transform for the configured
backbone/pretrained pair (resize/center-crop + CLIP's real per-channel
normalization stats), since the backbone is frozen and never fine-tuned here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.backbone import SharedBackbone

_DEFAULT_BACKBONE: SharedBackbone | None = None
_DEFAULT_CONFIG: dict | None = None

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load_and_preprocess(image_path: str, backbone: SharedBackbone) -> torch.Tensor:
    """Loads an image and applies open_clip's default preprocessing transform
    for `backbone`'s model/pretrained pair. Returns a (3, H, W) tensor.
    """
    img = Image.open(image_path).convert("RGB")
    return backbone.preprocess(img)


def extract_embeddings(
    image_paths: list[str],
    backbone: SharedBackbone,
    config: dict,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Returns an (N, embedding_dim) L2-normalized embedding matrix, one row
    per image in `image_paths` (same order).
    """
    backbone.eval().to(device)
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch = torch.stack([load_and_preprocess(p, backbone) for p in batch_paths])
            batch = batch.to(device)
            emb = backbone(batch).cpu().numpy()
            embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)
    # L2-normalize so cosine similarity == dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return embeddings / norms


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """embeddings must already be L2-normalized (see extract_embeddings)."""
    return embeddings @ embeddings.T


def _get_default_backbone(device: str = "cpu") -> tuple[SharedBackbone, dict]:
    """Lazily loads and caches the CLIP ViT-L/14 backbone from configs/config.yaml
    so repeated embed()/embed_folder() calls don't reload the model each time.
    """
    global _DEFAULT_BACKBONE, _DEFAULT_CONFIG
    if _DEFAULT_BACKBONE is None:
        from src.utils import load_config

        _DEFAULT_CONFIG = load_config("configs/config.yaml")
        _DEFAULT_BACKBONE = SharedBackbone(_DEFAULT_CONFIG).eval().to(device)
    return _DEFAULT_BACKBONE, _DEFAULT_CONFIG


def embed(image_path: str, device: str = "cpu") -> np.ndarray:
    """Embed a single image -> L2-normalized (embedding_dim,) vector
    (768-dim for the configured CLIP ViT-L/14 backbone).
    """
    backbone, config = _get_default_backbone(device)
    return extract_embeddings([image_path], backbone, config, device=device)[0]


def embed_folder(folder_path: str, device: str = "cpu", batch_size: int = 32) -> dict[str, np.ndarray]:
    """Embed every image in a folder -> {image_path: L2-normalized embedding}."""
    backbone, config = _get_default_backbone(device)
    image_paths = [
        str(p) for p in Path(folder_path).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    embeddings = extract_embeddings(image_paths, backbone, config, device=device, batch_size=batch_size)
    return dict(zip(image_paths, embeddings))
