"""Generate near-duplicate variants of AI images to serve as ground truth for
the similarity/clustering system (no public dataset labels "repetition", so
we construct our own test cases).

For each source image, produces 5-10 edited variants (crop/recolor/recompress/
resize) that SHOULD end up in the same similarity cluster. Variants from
different source images should NOT cluster together — that's the negative
control.

Usage:
    python src/data/near_duplicate_gen.py --input data/raw/sid_set/ai --out data/near_duplicates --n_variants 8
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.transforms import (
    apply_jpeg_compression,
    apply_color_jitter,
    apply_center_crop,
    apply_resize_roundtrip,
)
from src.utils import get_logger, ensure_dir, set_seed

logger = get_logger(__name__)

VARIANT_OPS = [
    lambda img: apply_jpeg_compression(img, random.choice([30, 50, 70])),
    lambda img: apply_color_jitter(img, 0.2, 0.2, 0.2),
    lambda img: apply_center_crop(img, random.uniform(0.7, 0.9)),
    lambda img: apply_resize_roundtrip(img, random.choice([0.5, 0.25])),
]


def make_variants(img: np.ndarray, n_variants: int) -> list[np.ndarray]:
    variants = []
    for _ in range(n_variants):
        out = img.copy()
        # apply 1-2 random ops per variant, simulating repost + light edit
        for op in random.sample(VARIANT_OPS, k=random.choice([1, 2])):
            out = op(out)
        variants.append(out)
    return variants


def main():
    parser = argparse.ArgumentParser(description="Build near-duplicate test set.")
    parser.add_argument("--input", required=True, help="Folder of source AI images.")
    parser.add_argument("--out", required=True, help="Output folder.")
    parser.add_argument("--n_source", type=int, default=50, help="Number of source images to sample.")
    parser.add_argument("--n_variants", type=int, default=8, help="Variants per source image.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    input_dir = Path(args.input)
    out_dir = ensure_dir(args.out)

    all_images = [p for p in input_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not all_images:
        logger.error("No images found in %s", input_dir)
        return

    sources = random.sample(all_images, min(args.n_source, len(all_images)))

    manifest = []  # ground-truth cluster labels: which variant came from which source
    for src_idx, src_path in enumerate(sources):
        cluster_id = f"cluster_{src_idx:04d}"
        cluster_dir = ensure_dir(out_dir / cluster_id)

        img = np.array(Image.open(src_path).convert("RGB"))
        Image.fromarray(img).save(cluster_dir / "original.jpg")
        manifest.append({"path": str(cluster_dir / "original.jpg"), "cluster_id": cluster_id, "is_original": True})

        variants = make_variants(img, args.n_variants)
        for v_idx, variant in enumerate(variants):
            v_path = cluster_dir / f"variant_{v_idx:02d}.jpg"
            Image.fromarray(variant).save(v_path)
            manifest.append({"path": str(v_path), "cluster_id": cluster_id, "is_original": False})

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Generated %d clusters (%d source images x ~%d variants each) -> %s",
        len(sources), len(sources), args.n_variants, out_dir,
    )
    logger.info("Ground truth cluster labels saved to %s", out_dir / "manifest.json")


if __name__ == "__main__":
    main()
