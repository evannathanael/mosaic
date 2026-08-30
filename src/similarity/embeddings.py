"""Extract embeddings from the SAME shared backbone used by the classifier —
no separate model, no extra training required. Cosine similarity between
these embeddings is what powers near-duplicate detection.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.backbone import SharedBackbone


def load_and_preprocess(image_path: str, backbone: SharedBackbone) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    return backbone.preprocess(img)  # real CLIP preprocessing, not an approximation


def extract_embeddings(
    image_paths: list[str],
    backbone: SharedBackbone,
    config: dict,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    backbone.eval().to(device)
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch = torch.stack([load_and_preprocess(p, backbone) for p in batch_paths]).to(device)
            emb = backbone(batch).cpu().numpy()
            embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return embeddings / norms


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """embeddings must already be L2-normalized (see extract_embeddings)."""
    return embeddings @ embeddings.T
