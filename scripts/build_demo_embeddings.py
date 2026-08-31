"""Precompute CLIP embeddings for every image served by the demo backend.

Writes ``data/dataset/embeddings.npz`` (paths relative to ``data/dataset/`` +
matching L2-normalized 768-d vectors). The backend loads this at startup
(backend/similarity_index.py) so seed posts have embeddings to match uploads
against, without paying the CLIP cost at request time.

Run once after changing the committed demo images (needs torch + open_clip;
downloads the CLIP ViT-L/14 weights on first run):

    python scripts/build_demo_embeddings.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.similarity.embed import IMAGE_EXTENSIONS, embed_folder  # noqa: E402

DATASET_DIR = ROOT / "data" / "dataset"
OUT = DATASET_DIR / "embeddings.npz"


def main() -> None:
    if not DATASET_DIR.exists():
        raise SystemExit(f"{DATASET_DIR} not found")

    embeddings = embed_folder(str(DATASET_DIR))
    paths, vecs = [], []
    for abs_path, vec in embeddings.items():
        rel = Path(abs_path).resolve().relative_to(DATASET_DIR).as_posix()
        paths.append(rel)
        vecs.append(np.asarray(vec, dtype="float32"))

    if not paths:
        raise SystemExit(f"No images ({', '.join(IMAGE_EXTENSIONS)}) under {DATASET_DIR}")

    np.savez_compressed(OUT, paths=np.array(paths), embeddings=np.stack(vecs))
    print(f"Wrote {len(paths)} embeddings -> {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
