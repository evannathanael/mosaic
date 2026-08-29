"""End-to-end inference script — THE required deliverable.

Takes an image directory, runs both the AI classifier and the similarity/
clustering system (sharing one backbone), and writes a JSON file with one
entry per image in the exact required format:

    {"image_path": "image_01.jpg", "pred": 0.91, "similarity_cluster": 4, "repetition_score": 0.94}

Usage:
    python src/inference.py --input_dir path/to/images --checkpoint outputs/baseline/model_best.pt --out outputs/predictions.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.models.classifier import AIGCDetector
from src.similarity.embeddings import extract_embeddings, load_and_preprocess
from src.similarity.clustering import compute_clusters, compute_repetition_scores
from src.utils import load_config, get_logger, ensure_dir

logger = get_logger(__name__)


def run_inference(input_dir: str, checkpoint_path: str, config: dict, device: str = "cpu") -> list[dict]:
    image_paths = sorted(
        str(p) for p in Path(input_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    if not image_paths:
        logger.error("No images found in %s", input_dir)
        return []

    logger.info("Running inference on %d images...", len(image_paths))
    model = AIGCDetector.load(checkpoint_path, config, device=device)

    # --- AI classification ---
    image_size = config["data"]["image_size"]
    preds = []
    batch_size = config["training"]["batch_size"]
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch = np.stack([load_and_preprocess(p, image_size) for p in batch_paths])
        batch_tensor = torch.from_numpy(batch).float().to(device)
        probs = model.predict_proba(batch_tensor).cpu().numpy()
        preds.extend(probs.tolist())

    # --- Similarity / near-duplicate clustering (reuses the SAME backbone) ---
    embeddings = extract_embeddings(image_paths, model.backbone, config, device=device)
    cluster_ids = compute_clusters(embeddings, config)
    repetition_scores = compute_repetition_scores(cluster_ids)

    results = []
    for path, pred, cluster_id, rep_score in zip(image_paths, preds, cluster_ids, repetition_scores):
        results.append({
            "image_path": path,
            "pred": round(float(pred), 4),
            "similarity_cluster": int(cluster_id),
            "repetition_score": round(float(rep_score), 4),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Run full Creator Balance inference.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/predictions.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_inference(args.input_dir, args.checkpoint, config, args.device)

    ensure_dir(str(Path(args.out).parent))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Wrote %d predictions to %s", len(results), args.out)


if __name__ == "__main__":
    main()
