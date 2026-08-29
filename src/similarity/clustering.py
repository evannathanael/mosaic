"""Group near-duplicate images using cosine similarity between shared-backbone
embeddings. Two clustering strategies are supported (set in config.yaml ->
similarity.clustering_method):

  - "threshold": simple, fast — connect any pair above `similarity.threshold`
    (via union-find), then take connected components as clusters. Good default.
  - "dbscan": more robust to noisy thresholds, using cosine distance.

Also computes the per-image `repetition_score` = (cluster size - 1) / total
images, i.e. how much of the corpus is made up of near-duplicates of this image.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score

from src.models.backbone import SharedBackbone
from src.similarity.embeddings import extract_embeddings, cosine_similarity_matrix
from src.utils import load_config, get_logger

logger = get_logger(__name__)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def cluster_threshold(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    n = embeddings.shape[0]
    sim = cosine_similarity_matrix(embeddings)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    unique_roots = {r: idx for idx, r in enumerate(sorted(set(roots)))}
    return np.array([unique_roots[r] for r in roots])


def cluster_dbscan(embeddings: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    # cosine distance = 1 - cosine similarity
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(embeddings)
    # DBSCAN labels noise points as -1; treat each as its own singleton cluster
    next_id = labels.max() + 1 if labels.max() >= 0 else 0
    out = labels.copy()
    for i, l in enumerate(labels):
        if l == -1:
            out[i] = next_id
            next_id += 1
    return out


def compute_clusters(embeddings: np.ndarray, config: dict) -> np.ndarray:
    sim_cfg = config["similarity"]
    if sim_cfg["clustering_method"] == "dbscan":
        return cluster_dbscan(embeddings, sim_cfg["dbscan_eps"], sim_cfg["dbscan_min_samples"])
    return cluster_threshold(embeddings, sim_cfg["threshold"])


def compute_repetition_scores(cluster_ids: np.ndarray) -> np.ndarray:
    """repetition_score[i] = (size of i's cluster - 1) / total images.
    0.0 = fully unique image, higher = part of a larger repeated group.
    """
    n = len(cluster_ids)
    counts = {}
    for c in cluster_ids:
        counts[c] = counts.get(c, 0) + 1
    return np.array([(counts[c] - 1) / n for c in cluster_ids])


def evaluate_clustering(image_paths: list[str], cluster_ids: np.ndarray, manifest_path: str) -> dict:
    """Compares predicted clusters against the ground-truth manifest produced
    by src/data/near_duplicate_gen.py. Reports pairwise precision/recall:
    for every pair of images, did we correctly predict "same cluster" vs
    "different cluster"?
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    truth = {m["path"]: m["cluster_id"] for m in manifest}

    path_to_idx = {p: i for i, p in enumerate(image_paths)}
    true_labels, pred_labels = [], []
    n = len(image_paths)
    for i in range(n):
        for j in range(i + 1, n):
            p_i, p_j = image_paths[i], image_paths[j]
            if p_i not in truth or p_j not in truth:
                continue
            true_same = int(truth[p_i] == truth[p_j])
            pred_same = int(cluster_ids[i] == cluster_ids[j])
            true_labels.append(true_same)
            pred_labels.append(pred_same)

    return {
        "pairwise_precision": precision_score(true_labels, pred_labels, zero_division=0),
        "pairwise_recall": recall_score(true_labels, pred_labels, zero_division=0),
        "n_pairs_evaluated": len(true_labels),
    }


def main():
    parser = argparse.ArgumentParser(description="Run and evaluate near-duplicate clustering.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input_dir", required=True, help="Folder of images (e.g. data/near_duplicates).")
    parser.add_argument("--manifest", default=None, help="Ground-truth manifest.json for evaluation.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    image_paths = [str(p) for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    logger.info("Found %d images in %s", len(image_paths), args.input_dir)

    backbone = SharedBackbone(config)
    embeddings = extract_embeddings(image_paths, backbone, config, device=args.device)
    cluster_ids = compute_clusters(embeddings, config)
    repetition_scores = compute_repetition_scores(cluster_ids)

    logger.info("Found %d clusters across %d images", len(set(cluster_ids)), len(image_paths))

    if args.manifest:
        metrics = evaluate_clustering(image_paths, cluster_ids, args.manifest)
        logger.info(
            "Clustering eval -> precision: %.3f, recall: %.3f (n_pairs=%d)",
            metrics["pairwise_precision"], metrics["pairwise_recall"], metrics["n_pairs_evaluated"],
        )


if __name__ == "__main__":
    main()
