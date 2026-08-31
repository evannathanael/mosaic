"""
Robustness curve for clustering: for each of the 6 official transforms, checks
whether a transformed image still matches its original above the configured
similarity threshold. Clustering equivalent of Glory's classifier robustness table.

Usage:
    python3 -m src.similarity.robustness_eval --input_dir data/raw/sid_set_sample \
        --config configs/config.yaml --n_images 30 --out outputs/clustering_robustness_table.csv
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.backbone import SharedBackbone
from src.data.transforms import NAMED_EVAL_TRANSFORMS
from src.utils import load_config, get_logger, ensure_dir

logger = get_logger(__name__)


def embed_image(img: Image.Image, backbone: SharedBackbone, device: str) -> np.ndarray:
    tensor = backbone.preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = backbone(tensor).cpu().numpy()[0]
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n_images", type=int, default=30)
    parser.add_argument("--out", default="outputs/clustering_robustness_table.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    config = load_config(args.config)
    threshold = config["similarity"]["threshold"]

    image_paths = [
        p for p in Path(args.input_dir).rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]
    image_paths = random.sample(image_paths, min(args.n_images, len(image_paths)))
    logger.info("Using %d source images, threshold=%.2f", len(image_paths), threshold)

    backbone = SharedBackbone(config)
    backbone.eval().to(args.device)

    results = []
    with torch.no_grad():
        for img_path in image_paths:
            orig_img = Image.open(img_path).convert("RGB")
            orig_arr = np.array(orig_img)
            clean_emb = embed_image(orig_img, backbone, args.device)

            for name, fn in NAMED_EVAL_TRANSFORMS.items():
                if name == "clean":
                    continue
                transformed_arr = fn(orig_arr.copy())
                transformed_img = Image.fromarray(transformed_arr)
                trans_emb = embed_image(transformed_img, backbone, args.device)

                sim = float(np.dot(clean_emb, trans_emb))
                results.append({
                    "transform": name,
                    "image": img_path.name,
                    "similarity": sim,
                    "still_clustered": sim >= threshold,
                })

    ensure_dir(Path(args.out).parent)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["transform", "image", "similarity", "still_clustered"])
        writer.writeheader()
        writer.writerows(results)

    agg = defaultdict(list)
    for r in results:
        agg[r["transform"]].append(r)

    print(f"{'transform':>15} | {'n':>4} | {'mean_sim':>9} | {'pct_still_clustered':>20}")
    for name, rows in agg.items():
        sims = [r["similarity"] for r in rows]
        pct = sum(r["still_clustered"] for r in rows) / len(rows)
        print(f"{name:>15} | {len(rows):>4} | {np.mean(sims):>9.3f} | {pct:>20.1%}")

    logger.info("Full results saved to %s", args.out)


if __name__ == "__main__":
    main()