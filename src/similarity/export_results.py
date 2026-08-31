"""
Runs the full pipeline (embed -> cluster -> repetition score) on a folder of
images and saves the deliverable output: one row per image with
similarity_cluster and repetition_score, for Evan/Eron to consume.

Usage:
    python3 -m src.similarity.export_results --input_dir data/raw/sid_set_sample \
        --config configs/config.yaml --out outputs/similarity_results.csv
"""
import argparse
import csv
from pathlib import Path

from src.models.backbone import SharedBackbone
from src.similarity.embed import extract_embeddings
from src.similarity.similarity import compute_clusters, compute_repetition_scores
from src.utils import load_config, get_logger, ensure_dir

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default="outputs/similarity_results.csv")
    parser.add_argument("--emb_out", default=None,
                        help="Optional .npz path to also dump the raw embedding matrix.")
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
    cluster_ids = compute_clusters(embeddings, config)
    repetition_scores = compute_repetition_scores(cluster_ids)

    ensure_dir(Path(args.out).parent)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "similarity_cluster", "repetition_score"])
        for path, cid, score in zip(image_paths, cluster_ids, repetition_scores):
            writer.writerow([path, int(cid), round(float(score), 6)])

    if args.emb_out:
        import numpy as np

        ensure_dir(Path(args.emb_out).parent)
        np.savez_compressed(args.emb_out, paths=np.array(image_paths), embeddings=embeddings)
        logger.info("Saved embedding matrix %s to %s", embeddings.shape, args.emb_out)

    logger.info("Saved %d rows to %s", len(image_paths), args.out)
    logger.info(
        "Clusters: %d | repetition_score range: %.4f - %.4f",
        len(set(cluster_ids)), min(repetition_scores), max(repetition_scores),
    )


if __name__ == "__main__":
    main()