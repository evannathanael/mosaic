"""Extract embeddings from the SAME shared backbone used by the classifier —
no separate model, no extra training required. Cosine similarity between
these embeddings is what powers near-duplicate detection.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.backbone import SharedBackbone


def load_and_preprocess(image_path: str, image_size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)  # CHW


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
    image_size = config["data"]["image_size"]
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch = np.stack([load_and_preprocess(p, image_size) for p in batch_paths])
            batch_tensor = torch.from_numpy(batch).float().to(device)
            emb = backbone(batch_tensor).cpu().numpy()
            embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)
    # L2-normalize so cosine similarity == dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return embeddings / norms


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """embeddings must already be L2-normalized (see extract_embeddings)."""
    return embeddings @ embeddings.T
