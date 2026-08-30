"""
Sweep the similarity threshold against the ground-truth manifest to find
where precision/recall actually trade off, instead of trusting the config's
untuned default.

Usage:
    python3 -m src.similarity.threshold_sweep \
        --input_dir data/near_duplicates \
        --config configs/config.yaml \
        --manifest data/near_duplicates/manifest.json
"""
import argparse
from pathlib import Path

import numpy as np

from src.models.backbone import SharedBackbone
from src.similarity.embeddings import extract_embeddings, cosine_similarity_matrix
from src.similarity.clustering import cluster_threshold, evaluate_clustering
from src.utils import load_config, get_logger

logger = get_logger(__name__)

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    image_paths = [
        str(p) for p in Path(args.input_dir).rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]
    logger.info("Found %d images", len(image_paths))

    backbone = SharedBackbone(config)
    embeddings = extract_embeddings(image_paths, backbone, config, device=args.device)

    sim = cosine_similarity_matrix(embeddings)
    off_diag = sim[~np.eye(sim.shape[0], dtype=bool)]
    logger.info(
        "Pairwise similarity stats -> min: %.3f, median: %.3f, mean: %.3f, max: %.3f",
        off_diag.min(), np.median(off_diag), off_diag.mean(), off_diag.max(),
    )

    print(f"{'threshold':>10} | {'n_clusters':>10} | {'precision':>9} | {'recall':>7}")
    for t in THRESHOLDS:
        cluster_ids = cluster_threshold(embeddings, t)
        metrics = evaluate_clustering(image_paths, cluster_ids, args.manifest)
        print(
            f"{t:>10.2f} | {len(set(cluster_ids)):>10} | "
            f"{metrics['pairwise_precision']:>9.3f} | {metrics['pairwise_recall']:>7.3f}"
        )


if __name__ == "__main__":
    main()